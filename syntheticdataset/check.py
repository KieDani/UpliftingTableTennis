import numpy as np
import mujoco
import os
import matplotlib.pyplot as plt

from syntheticdataset.mujocosimulation import _count_hits, TABLE_HEIGHT, TABLE_LENGTH, TABLE_WIDTH, NET_HEIGHT, Net_WIDTH, MAX_TIME, TIMESTEP
from syntheticdataset.mujocosimulation import _calc_cammatrices, world2cam, cam2img, FPS, CAMERA, WIDTH, HEIGHT, XML, _init_simulation
from syntheticdataset.helper import table_connections, table_lines, table_points



def _init_deteministic_(r, v, mode, direction):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # initial velocity: 3 < |v| < 35, 110° < phi < 250°, 10° < theta < 170°
    if 'first' in mode:  # point to the center of the according half of the players table
        c_y = TABLE_WIDTH / 2 if r[1] > 0 else -TABLE_WIDTH / 2
        c_x = TABLE_LENGTH / 2 if direction == 'left_to_right' else -TABLE_LENGTH / 2
        center_opponent = np.array([c_x, c_y, TABLE_HEIGHT])
    else:  # point to the center of the opponents table
        c_x = -TABLE_LENGTH / 2 if direction == 'left_to_right' else TABLE_LENGTH / 2
        center_opponent = np.array([c_x, 0, TABLE_HEIGHT])
    # TODO: is this correct for right_to_left?

    w = np.array([0, 0, 0])  # no initial angular velocity

    # set initial position and velocity
    data.qpos[0:3] = r
    data.qvel[0:3] = v
    data.qvel[3:6] = w

    return model, data


def simulate_trajectory(model, data, mode, direction):
    # renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    mujoco.mj_step(model, data)
    # renderer.update_scene(data, camera=CAMERA)

    positions, velocities, rotations = [], [], []
    ex_mats = []
    in_mats = []
    times = []
    next_save_time = 0.
    while next_save_time < MAX_TIME:
        steps = round((next_save_time - data.time) / TIMESTEP)
        mujoco.mj_step(model, data, steps)

        correct_side = data.qpos[0] < 0 if direction == 'left_to_right' else data.qpos[0] > 0  # check if ball is on opponents side
        # check if ball is out of bounds
        if mode == 'final_lose':  # no bounce
            if abs(data.qpos[0]) > 6 or abs(data.qpos[1]) > 3:
                break
        elif mode == 'final_win':  # two bounces on opponent side
            if correct_side and (abs(data.qpos[0]) > 1.38 or abs(data.qpos[1]) > 0.77 or data.qpos[2] < 0.7):
                break
        elif mode == 'intermediate':  # one bounce on opponent side
            if correct_side and (abs(data.qpos[0]) > 4.5 or abs(data.qpos[1]) > 2.5):
                break
        elif mode == 'first_good':  # first bounce on players side, second bounce on opponent side
            if correct_side and (abs(data.qpos[0]) > 2.5 or abs(data.qpos[1]) > 1.5):
                break
        elif mode == 'first_short':  # first bounce on players side, second bounce on players side
            if abs(data.qpos[0]) > 2.5 or abs(data.qpos[1]) > 1.5 or data.qpos[2] < 0.5:
                break
        elif mode == 'first_long':  # one bounce on players side
            if correct_side and abs(data.qpos[0]) > 2.5 or abs(data.qpos[1]) > 1.5:
                break

        # check if ball is still in the image plane
        # renderer.update_scene(data, camera=CAMERA)
        ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
        r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
        r_img = cam2img(r_cam, in_mat[:3, :3])
        if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))):
            break

        positions.append(data.qpos[0:3].copy())
        velocities.append(data.qvel[0:3].copy())
        rotations.append(data.qvel[3:6].copy())
        times.append(next_save_time)
        ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA)
        in_mat = in_mat[:3, :3]
        ex_mats.append(ex_mat)
        in_mats.append(in_mat)
        next_save_time += 1 / FPS

    # check if list is empty -> I do the check later, but it is also needed now because the following code would fail otherwise
    minimum_length = int(round(0.2 * FPS))  # less than 0.2 seconds is unreasonable
    if len(positions) < minimum_length:
        print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")

    # calculate the number of bounces
    hits_opponent, hits_own, hits_ground = _count_hits(positions, direction)
    # check maximum height
    max_height = 1.4 if 'first' in mode else 1.8
    if np.max(np.array(positions)[:, 2]) > max_height:
        print(f"Trajectory too high: {np.max(np.array(positions)[:, 2])} m, mode: {mode}, direction: {direction}")

    min_percent = 0.2  # minimum percentage of time before cutting for a trajectory to be valid
    # cut trajectory
    if mode == 'final_lose':  # no bounce
        if len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_ground = []
    elif mode == 'final_win':  # two bounces on opponent side
        if len(hits_opponent) > 2:
            cut_time = hits_opponent[2]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_opponent = hits_opponent[:2]
            hits_ground = []
        elif len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_ground = []
    elif mode == 'intermediate':  # one bounce on opponent side
        if len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_ground = []
    elif mode == 'first_good':  # first bounce on players side, second bounce on opponent side
        if len(hits_opponent) > 1:
            cut_time = hits_opponent[1]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_opponent = hits_opponent[:1]
            hits_ground = []
        elif len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_ground = []
    elif mode == 'first_short':  # first bounce on players side, second bounce on players side
        if len(hits_own) > 2:
            cut_time = hits_own[2]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_own = hits_own[:2]
            hits_opponent = []
            hits_ground = []
        elif len(hits_opponent) > 0:
            cut_time = hits_opponent[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_opponent = []
            hits_ground = []
        elif len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_ground = []
    elif mode == 'first_long':  # one bounce on players side
        if len(hits_ground) > 0:
            cut_time = hits_ground[0]
            if cut_time < min_percent * MAX_TIME:
                print(f"Trajectory too short: {len(positions)} frames, mode: {mode}, direction: {direction}")
            cut_index = np.sum(np.where(np.array(times) < cut_time, 1, 0)) - 1
            positions, velocities, rotations, times = positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index]
            hits_ground = []

    # ensure minimum length of 7 frames
    if len(positions) < minimum_length:
        print(f"Trajectory too short after cutting: {len(positions)} frames, mode: {mode}, direction: {direction}")

    # check if ball is above the net
    heights_close_to_net = np.array(positions)[:, 2][np.abs(np.array(positions)[:, 0]) < 0.04]
    widths_close_to_net = np.array(positions)[:, 1][np.abs(np.array(positions)[:, 0]) < 0.04]
    if len(heights_close_to_net) > 0 and np.max(heights_close_to_net) < NET_HEIGHT and np.min(np.abs(widths_close_to_net)) < Net_WIDTH / 2:
        print(f"Trajectory too low: max height {np.max(heights_close_to_net)} m, mode: {mode}, direction: {direction}")

    # check if final ball position is on the correct side
    is_opposite_site = lambda x: x < 0 if direction == 'left_to_right' else x > 0
    if mode == 'final_lose':
        if not is_opposite_site(positions[-1][0]):
            print(f"Final position not on opponent side: {positions[-1][0]}, mode: {mode}, direction: {direction}")
    elif mode == 'first_long':
        if not is_opposite_site(positions[-1][0]):
            print(f"Final position not on opponent side: {positions[-1][0]}, mode: {mode}, direction: {direction}")
    # maybe add first_short -> should it be on players side? np.max(positions[:, 0]) < 0

    # check if number of bounces is fitting to the mode
    if mode == 'final_lose':  # no bounce
        if len(hits_opponent) == 0 and len(hits_own) == 0 and len(hits_ground) == 0:
            pass
        else:
            print(f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}")
    elif mode == 'final_win':  # two bounces on opponent side
        if len(hits_opponent) == 2 and len(hits_own) == 0 and len(hits_ground) == 0:
            pass
        else:
            print(f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}")
    elif mode == 'intermediate':  # one bounce on opponent side
        if len(hits_opponent) == 1 and len(hits_own) == 0 and len(hits_ground) == 0:
            pass
        else:
            print(f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}")
    elif mode == 'first_good':  # first bounce on players side, second bounce on opponent side
        if len(hits_opponent) == 1 and len(hits_own) == 1 and len(hits_ground) == 0:
            pass
        else:
            print(f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}")
    elif mode == 'first_short':  # first bounce on players side, second bounce on players side
        if len(hits_opponent) == 0 and len(hits_own) == 2 and len(hits_ground) == 0:
            pass
        else:
            print(f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}")
    elif mode == 'first_long':  # one bounce on players side
        if len(hits_opponent) == 0 and len(hits_own) == 1 and len(hits_ground) == 0:
            pass
        else:
            print(f"Unexpected number of bounces: {len(hits_opponent)} opponent, {len(hits_own)} own, {len(hits_ground)} ground, mode: {mode}, direction: {direction}")

    return np.array(positions), np.array(velocities), np.array(rotations), np.array(times), np.array(ex_mats), np.array(in_mats)


def plot_imagecoords(r_gt, r_pred, table_img, title=''):
    '''Plot the image coordinates of original and regressed trajectory + plot the table points in the image plane.
    Args:
        r_gt (np.array): Ground truth image coordinates
        r_pred (np.array): Predicted image coordinates
        table_img (np.array): Table points in image coordinates
    '''
    # draw lines between the table points
    for connection in table_connections:
        plt.plot(table_img[connection, 0], table_img[connection, 1], 'k')
    # draw ground truth trajectory
    plt.plot(r_gt[:, 0], r_gt[:, 1], 'r', label='Ground truth')
    # draw regressed trajectory
    plt.plot(r_pred[:, 0], r_pred[:, 1], 'b--', label='Predicted')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, WIDTH)
    plt.ylim(HEIGHT, 0)
    #plt.title(title)
    plt.legend()
    plt.show()



def main():
    mode = 'first_good'
    seed = 7

    # left to right
    direction = 'left_to_right'

    r0 = np.array([2, 0, NET_HEIGHT + 0.4])  # initial position
    v0 = np.array([-5, 0, 0])  # initial velocity
    model, data = _init_deteministic_(r0, v0, mode, direction)
    # model, data = _init_simulation(seed, mode, direction)
    # print(data.qpos[0:3], data.qvel[0:3], data.qvel[3:6])  # print initial position and velocity

    positions, velocities, rotations, times, ex_mats, in_mats = simulate_trajectory(model, data, mode, direction)
    # plot the image coordinates
    r_cam = world2cam(positions, ex_mats[0])
    r_img = cam2img(r_cam, in_mats[0][:3, :3])
    table_points_img = cam2img(world2cam(table_points, ex_mats[0]), in_mats[0][:3, :3])
    plot_imagecoords(r_img, r_img, table_points_img, title=f'{mode} - {direction}')

    # right to left
    direction = 'right_to_left'

    r0 = np.array([-2, 0, NET_HEIGHT + 0.4])  # initial position
    v0 = np.array([5, 0, 0])  # initial velocity
    model, data = _init_deteministic_(r0, v0, mode, direction)
    # model, data = _init_simulation(seed, mode, direction)
    # print(data.qpos[0:3], data.qvel[0:3], data.qvel[3:6])  # print initial position and velocity

    positions_r, velocities_r, rotations_r, times_r, ex_mats_r, in_mats_r = simulate_trajectory(model, data, mode, direction)
    # plot the image coordinates
    r_cam_r = world2cam(positions_r, ex_mats_r[0])
    r_img_r = cam2img(r_cam_r, in_mats_r[0][:3, :3])
    table_points_img_r = cam2img(world2cam(table_points, ex_mats_r[0]), in_mats_r[0][:3, :3])
    plot_imagecoords(r_img_r, r_img_r, table_points_img_r, title=f'{mode} - {direction}')

    pass


if __name__ == '__main__':
    main()