import numpy as np
import einops as eo
import torch
import mujoco
import mujoco_viewer

# --- Constants ---

# -------------------------------------
# --- Physical & Simulation Globals ---
# -------------------------------------

# Screen and Camera Configuration
HEIGHT, WIDTH = 1080, 1920
CAMERA_NAME = 'main' # The name of the camera defined in the XML.

# Table Tennis Table Dimensions (in meters, following official ITTF rules)
TABLE_HEIGHT = 0.76
TABLE_WIDTH = 1.525
TABLE_LENGTH = 2.74
NET_POST_OFFSET = 0.1525         # How far the net posts extend from the table sides.
NET_HEIGHT_ABOVE_TABLE = 0.1525
NET_TOTAL_HEIGHT = TABLE_HEIGHT + NET_HEIGHT_ABOVE_TABLE
NET_TOTAL_WIDTH = TABLE_WIDTH + 2 * NET_POST_OFFSET

# MuJoCo Simulation Parameters
TIMESTEP = 0.001                 # Simulation timestep in seconds. Smaller values increase accuracy but are slower.
MAX_SIMULATION_TIME = 1.0        # Maximum duration of a single trajectory simulation.
FPS = 500                        # Frames per second for saving the trajectory data.

# -------------------------------------
# --- Trajectory Analysis Constants ---
# -------------------------------------

# Parameters for Hit Detection in the `_count_hits` function
HIT_DETECTION_Z_THRESHOLD_TABLE = TABLE_HEIGHT + 0.04 # Z-height threshold to robustly detect a table bounce.
HIT_DETECTION_Z_THRESHOLD_GROUND = 0.08              # Z-height threshold to detect a ground bounce.
HIT_DETECTION_X_MARGIN = 0.01                        # A small margin to avoid detecting hits exactly at the net line (x=0).
HIT_TIME_INTERPOLATION_WEIGHTS = (0.75, 0.25)          # Weights for interpolating the exact hit time, combining the interval midpoint and the point of minimum height.

# -------------------------------------
# --- Camera & Visualization Setup ---
# -------------------------------------

# Camera Pose and Intrinsics
# These values define the camera's position, orientation, and lens properties.
fx, fy = 2033, 2180
camera_pos = np.array([0.04381194, 8.92938715, 5.40070126])
camera_up = np.array([7.81340900e-04, -4.33644716e-01, 9.01083598e-01])
camera_right = np.array([-0.99998599, 0.00437903, 0.0029745])

# Pre-calculated Table Geometry for visualization purposes
table_points = np.array([
    [-TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT],  # 0 close left
    [-TABLE_LENGTH / 2, -TABLE_WIDTH / 2, TABLE_HEIGHT],  # 1 close right
    [0.0, TABLE_WIDTH / 2, TABLE_HEIGHT],  # 2 center left
    [0.0, -TABLE_WIDTH / 2, TABLE_HEIGHT],  # 3 center right
    [TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT],  # 4 far left
    [TABLE_LENGTH / 2, -TABLE_WIDTH / 2, TABLE_HEIGHT],  # 5 far right
    [0.0, TABLE_WIDTH / 2 + NET_POST_OFFSET, TABLE_HEIGHT],  # 6 net left bottom
    [0.0, -(TABLE_WIDTH / 2 + NET_POST_OFFSET), TABLE_HEIGHT],  # 7 net right bottom
    [0.0, 0.0, TABLE_HEIGHT],  # 8 net center bottom
    [0.0, TABLE_WIDTH / 2 + NET_POST_OFFSET, NET_TOTAL_HEIGHT],  # 9 net left top
    [0.0, -(TABLE_WIDTH / 2 + NET_POST_OFFSET), NET_TOTAL_HEIGHT],  # 10 net right top
    [-TABLE_LENGTH / 2, 0, TABLE_HEIGHT],  # 11 close center
    [TABLE_LENGTH / 2, 0, TABLE_HEIGHT],  # 12 far center
])
table_connections = [
    (0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (4, 5),
    (6, 2), (2, 3), (3, 7), (6, 9), (7, 10), (9, 10),
    (11, 8), (12, 8),
]
table_lines = [
    [0, 2, 4], [1, 3, 5], [11, 8, 12], [0, 11, 1],
    [4, 12, 5], [6, 8, 7], [9, 10], [6, 9], [7, 10],
]

# --- MuJoCo XML Definition ---
XML = f"""
<mujoco>
  <option cone="elliptic" gravity="0 0 -9.81" integrator="implicit" timestep="{TIMESTEP}" density="1.225" viscosity="0.000018" />
  <asset>
    <material name="ball_material" reflectance="0" rgba="1 1 1 1"/>
    <texture name="table_texture" type="cube" filefront="syntheticdataset/table.png" fileup="syntheticdataset/black.png" filedown="syntheticdataset/black.png" fileleft="syntheticdataset/black.png" fileright="syntheticdataset/black.png" />
    <material name="table_material" reflectance=".2" texture="table_texture" texuniform="false"/>
    <texture name="net_texture" type="cube" fileleft="syntheticdataset/net.png" fileright="syntheticdataset/net.png" />
    <material name="net_material" reflectance="0" texture="net_texture" texuniform="false" texrepeat="1 1" />
  </asset>
  <visual>
    <global offheight="{HEIGHT}" offwidth="{WIDTH}"/>
  </visual>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="-2 -2 10" dir="0.1 0.1 -1"/>
    <geom type="plane" size="800 800 0.1" rgba="0.6 0.6 0.6 1" pos="0 0 0" material="ball_material"/>
    <body name="ball_body" pos="0 0 1.2">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size=".02" material="ball_material" mass=".0027" fluidshape="ellipsoid" fluidcoef="0.235 0.25 0.0 1.0 1.0"/>
    </body>
    <geom name="table_geom" type="box" pos="0 0 {TABLE_HEIGHT/2}" size="{TABLE_LENGTH/2} {TABLE_WIDTH/2} {TABLE_HEIGHT/2}" material="table_material"/>
    <geom name="net_geom" type="box" pos="0 0 {TABLE_HEIGHT}" size="0.02 {TABLE_HEIGHT+NET_POST_OFFSET} {NET_HEIGHT_ABOVE_TABLE}" material="net_material" rgba="1 1 1 0.6" />
    <camera name="{CAMERA_NAME}"
            focal="{fx/WIDTH} {fy/HEIGHT}"
            resolution="{WIDTH} {HEIGHT}"
            sensorsize="1 1"
            pos="{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}"
            mode="fixed"
            xyaxes="{camera_right[0]} {camera_right[1]} {camera_right[2]} {camera_up[0]} {camera_up[1]} {camera_up[2]}"/>
  </worldbody>
  <default>
    <pair solref="-1000000 -17" solreffriction="-0.0 -200.0" friction="0.1 0.1 0.005 0.0001 0.0001" solimp="0.98 0.99 0.001 0.5 2"/>
  </default>
  <contact>
    <pair geom1="ball_geom" geom2="table_geom"/>
    <pair geom1="ball_geom" geom2="net_geom"/>
  </contact>
</mujoco>
"""

# --- Utility Functions (Unchanged) ---

def get_cameralocations(Mexts):
    '''Get the camera location from the extrinsic matrix.'''
    if len(Mexts.shape) == 3:
        R_transposed = eo.rearrange(Mexts[:, :3, :3], 't i j -> t j i') # R^-1 = R^T
        c = np.einsum('t i j, t j -> t i', -R_transposed, Mexts[:, :3, 3]) # R^-1 * -t
    elif len(Mexts.shape) == 2:
        R_transposed = Mexts[:3, :3].T
        c = -R_transposed @ Mexts[:3, 3]
    else:
        raise ValueError('Shape not supported.')
    return c

def get_forwards(Mexts):
    '''Get the normalized forward direction from the extrinsic matrix.'''
    forwards = Mexts[..., 2, :3]
    forwards /= np.linalg.norm(forwards, axis=-1)[..., np.newaxis]
    return forwards

def get_ups(Mexts):
    '''Get the normalized up direction from the extrinsic matrix.'''
    ups = -Mexts[..., 1, :3]
    ups /= np.linalg.norm(ups, axis=-1)[..., np.newaxis]
    return ups

def get_rights(Mexts):
    '''Get the normalized right direction from the extrinsic matrix.'''
    rights = Mexts[..., 0, :3]
    rights /= np.linalg.norm(rights, axis=-1)[..., np.newaxis]
    return rights

def get_Mext(c, f, r):
    '''Get the extrinsic matrix from the camera location, forward and right directions.'''
    if len(c.shape) == len(f.shape) == len(r.shape) == 1:
        up = np.cross(f, r)
        up /= np.linalg.norm(up)
        R = np.zeros((3, 3))
        R[0, :] = r
        R[1, :] = up
        R[2, :] = f
        t = -R @ c
        Mext = np.eye(4)
        Mext[:3, :3] = R
        Mext[:3, 3] = t
        return Mext
    elif len(c.shape) == len(f.shape) == len(r.shape) == 2:
        up = np.cross(f, r)
        up /= np.linalg.norm(up, axis=1)[:, np.newaxis]
        R = np.zeros((len(c), 3, 3))
        R[:, 0, :] = r
        R[:, 1, :] = up
        R[:, 2, :] = f
        t = -np.einsum('t i j, t j -> t i', R, c)
        Mext = np.zeros((len(c), 4, 4))
        Mext[:, :3, :3] = R
        Mext[:, :3, 3] = t
        Mext[:, 3, 3] = 1
        return Mext
    else:
        raise ValueError('Shape not supported.')

def cam2img(r_cam, Mints):
    '''Project a batch of 3D points to image coordinates.'''
    if len(r_cam.shape) == 1:
        if len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, j -> i')
            r_img = r_img[:2] / r_img[2]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_cam.shape) == 2:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, 'b i j, b j -> b i')
            r_img = r_img[:, :2] / r_img[:, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, b j -> b i')
            r_img = r_img[:, :2] / r_img[:, 2:3]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_cam.shape) == 3:
        if len(Mints.shape) == 3:
            r_img = eo.einsum(Mints[:, :3, :3], r_cam, 'b i j, b t j -> b t i')
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        elif len(Mints.shape) == 2:
            r_img = eo.einsum(Mints[:3, :3], r_cam, 'i j, b t j -> b t i')
            r_img = r_img[:, :, :2] / r_img[:, :, 2:3]
        else:
            raise ValueError('Shape not supported.')
    else:
        raise ValueError('Shape not supported.')
    return r_img

def world2cam(r_world, Mexts):
    '''Transform a batch of 3D points from world to camera coordinates.'''
    if len(r_world.shape) == 1:
        D = r_world.shape
        if len(Mexts.shape) == 2:
            r_world = concat(r_world, (D,))
            r_cam = eo.einsum(Mexts, r_world, 'i j, j -> i')
            r_cam = r_cam[:3] / r_cam[3]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_world.shape) == 2:
        T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, 'b i j, b j -> b i')
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (T, D))
            r_cam = eo.einsum(Mexts, r_world, 'i j, b j -> b i')
            r_cam = r_cam[:, :3] / r_cam[:, 3:4]
        else:
            raise ValueError('Shape not supported.')
    elif len(r_world.shape) == 3:
        B, T, D = r_world.shape
        if len(Mexts.shape) == 3:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, 'b i j, b t j -> b t i')
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        elif len(Mexts.shape) == 2:
            r_world = concat(r_world, (B, T, D))
            r_cam = eo.einsum(Mexts, r_world, 'i j, b t j -> b t i')
            r_cam = r_cam[:, :, :3] / r_cam[:, :, 3:4]
        else:
            raise ValueError('Shape not supported.')
    else:
        raise ValueError('Shape not supported.')
    return r_cam

def concat(x, shape):
    """
    Concatenates a tensor `x` with a tensor of ones along the last dimension.
    """
    if isinstance(x, np.ndarray):
        ones = np.ones((*shape[:-1], 1))
        return np.concatenate([x, ones], axis=-1)
    elif isinstance(x, torch.Tensor):
        ones = torch.ones((*shape[:-1], 1), device=x.device)
        return torch.cat([x, ones], dim=-1)
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor.")

def _calc_cammatrices(data, camera_name):
    # (Unchanged from original)
    camera_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = eo.rearrange(data.cam_xmat[camera_id], '(i j) -> i j', i=3, j=3).T
    R = np.eye(3)
    R[0, :] = weird_R[0, :]
    R[1, :] = - weird_R[1, :]
    R[2, :] = - weird_R[2, :]
    cam_pos = data.cam_xpos[camera_id]
    t = -np.dot(R, cam_pos)
    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R
    ex_mat[:3, 3] = t
    fx = data.model.cam_intrinsic[camera_id][0] / data.model.cam_sensorsize[camera_id][0] * data.model.cam_resolution[camera_id][0]
    fy = data.model.cam_intrinsic[camera_id][1] / data.model.cam_sensorsize[camera_id][1] * data.model.cam_resolution[camera_id][1]
    cx = (WIDTH - 1) / 2
    cy = (HEIGHT - 1) / 2
    in_mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1], [0, 0, 0]])
    return ex_mat, in_mat

def _count_hits(positions, direction):
    """
    Counts bounces on own side, opponent side, and ground.
    Updated to use constants instead of magic numbers.
    """
    hits_own = []
    hits_opponent = []
    hits_ground = []

    opposite_side_fn = lambda x: -HIT_DETECTION_X_MARGIN > x > -TABLE_LENGTH / 2 if direction == 'left_to_right' else TABLE_LENGTH / 2 > x > HIT_DETECTION_X_MARGIN
    own_side_fn = lambda x: TABLE_LENGTH / 2 > x > HIT_DETECTION_X_MARGIN if direction == 'left_to_right' else -HIT_DETECTION_X_MARGIN > x > -TABLE_LENGTH / 2

    binary_mask_z = np.array([pos[2] < HIT_DETECTION_Z_THRESHOLD_TABLE for pos in positions])
    binary_mask_y = np.array([abs(pos[1]) < TABLE_WIDTH / 2 for pos in positions])
    binary_mask_x_opponent = np.array([opposite_side_fn(pos[0]) for pos in positions])
    binary_mask_x_own = np.array([own_side_fn(pos[0]) for pos in positions])
    binary_mask_opponent = binary_mask_z & binary_mask_y & binary_mask_x_opponent
    binary_mask_own = binary_mask_z & binary_mask_y & binary_mask_x_own
    binary_mask_ground = np.array([pos[2] <= HIT_DETECTION_Z_THRESHOLD_GROUND for pos in positions])

    positions = np.array(positions)
    w1, w2 = HIT_TIME_INTERPOLATION_WEIGHTS

    for mask, hit_list in [(binary_mask_opponent, hits_opponent),
                           (binary_mask_own, hits_own),
                           (binary_mask_ground, hits_ground)]:
        start, end = None, None
        for i, b in enumerate(mask):
            if i == 0 and b:
                start = i
            elif b and not mask[i - 1]:
                start = i
            if not b and mask[i - 1] and i != 0:
                end = i - 1
                # Interpolate hit time based on a weighted average of the interval midpoint and the time of minimum height
                mid_point_time = (end + start) / 2 / FPS
                min_height_time = (np.argmin(positions[start:end + 1, 2]) + start) / FPS
                hit_list.append(w1 * mid_point_time + w2 * min_height_time)

    return hits_opponent, hits_own, hits_ground

def test_visualization(model, data):
    """Init syntheticdataset and show it with mujoco viewer"""
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 5
    for _ in range(10000):
        if viewer.is_alive:
            mujoco.mj_step(model, data)
            if _ % 50 == 0:
                viewer.render()
        else:
            break
    viewer.close()

if __name__ == '__main__':
    pass