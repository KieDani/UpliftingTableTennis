if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate inference dataset')
    parser.add_argument('--num_trajectories', type=int, default=50000, help='Number of trajectories to generate')
    parser.add_argument('--folder', type=str, default='tmp', help='Folder to save the dataset')
    parser.add_argument('--num_processes', type=int, default=128, help='Number of cpu processes to use')
    parser.add_argument('--mode', type=str, default='intermediate', help='Mode of the syntheticdataset, e.g. intermediate, final, first')
    parser.add_argument('--direction', type=str, default='left_to_right', help='Direction of the syntheticdataset, e.g. left_to_right, right_to_left')
    args = parser.parse_args()

import mujoco
import numpy as np
import random
import math
import os
import tqdm
from multiprocessing import Pool

# Import from the new refactored helper file
import syntheticdataset.helper as helper
from syntheticdataset.helper import (
    cam2img, world2cam, _calc_cammatrices, _count_hits, XML,
    HEIGHT, WIDTH, TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT, NET_TOTAL_HEIGHT, NET_TOTAL_WIDTH,
    TIMESTEP, MAX_SIMULATION_TIME, FPS, CAMERA_NAME
)

# --- Constants for Trajectory Generation and Validation ---

INIT_POS_RANGES = {
    'first': {'x': (1.0, 2.5), 'y': (-1.5, 1.5), 'z': (0.8, 1.6)},
    'other': {'x': (0.1, 4.0), 'y': (-2.0, 2.0), 'z_table_clear': (0.8, 1.8), 'z_other': (0.5, 1.8)}
}
INIT_VEL_SPEED_RANGE = (3.0, 30.0)
INIT_VEL_PHI_DEVIATION_DEG = 60.0
INIT_VEL_THETA_DEVIATION_DEG = {'below': (25.0, 60.0), 'above': (25.0, 60.0)}
INIT_ANG_VEL_SPEED_RANGE = (0.0, 500.0)
MIN_TRAJ_DURATION_SEC = 0.2
MIN_TRAJ_LEN_FRAMES = int(round(MIN_TRAJ_DURATION_SEC * FPS))
MIN_TRAJ_CUT_TIME_RATIO = 0.2
MAX_HEIGHT_FIRST_MODE = 1.4
MAX_HEIGHT_OTHER_MODES = 1.8
NET_CLEARANCE_X_MARGIN = 0.04
OOB_DEFINITIONS = {
    'final_lose': (6.0, 3.0, -1.0),
    'final_win': (TABLE_LENGTH / 2, TABLE_WIDTH, 0.7),
    'intermediate': (4.5, 2.5, -1.0),
    'first_good': (2.5, 1.5, -1.0),
    'first_short': (2.5, 1.5, 0.5),
    'first_long': (2.5, 1.5, -1.0),
}


def _init_simulation(seed, mode, direction):
    rng_py = random.Random(seed)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    sign_x = 1 if direction == 'left_to_right' else -1
    r = np.empty(3, dtype=np.float64)
    if 'first' in mode:
        ranges = INIT_POS_RANGES['first']
        r[0] = rng_py.uniform(*ranges['x']) * sign_x;
        r[1] = rng_py.uniform(*ranges['y']);
        r[2] = rng_py.uniform(*ranges['z'])
    else:
        ranges = INIT_POS_RANGES['other']
        r[0] = rng_py.uniform(*ranges['x']) * sign_x;
        r[1] = rng_py.uniform(*ranges['y'])
        if abs(r[0]) < TABLE_LENGTH / 2 and abs(r[1]) < TABLE_WIDTH / 2:
            r[2] = rng_py.uniform(*ranges['z_table_clear'])
        else:
            r[2] = rng_py.uniform(*ranges['z_other'])
    if 'first' in mode:
        c_y = TABLE_WIDTH / 2 if r[1] > 0 else -TABLE_WIDTH / 2
        c_x = TABLE_LENGTH / 2 if direction == 'left_to_right' else -TABLE_LENGTH / 2
        center_opponent = np.array([c_x, c_y, TABLE_HEIGHT])
    else:
        c_x = -TABLE_LENGTH / 2 if direction == 'left_to_right' else TABLE_LENGTH / 2
        center_opponent = np.array([c_x, 0, TABLE_HEIGHT])
    angle_opponent_phi = np.rad2deg(math.atan2(r[1] - center_opponent[1], r[0] - center_opponent[0]));
    base_phi = 180 + angle_opponent_phi
    angle_opponent_theta = np.rad2deg(math.atan2(r[2] - center_opponent[2], abs(r[0] - center_opponent[0])));
    base_theta = 90 - angle_opponent_theta

    if r[2] < center_opponent[2]:
        min_theta = max(90.0, base_theta - INIT_VEL_THETA_DEVIATION_DEG['below'][0])
        max_theta = min(170.0, base_theta + INIT_VEL_THETA_DEVIATION_DEG['below'][1])
    else:
        min_theta = max(10.0, base_theta - INIT_VEL_THETA_DEVIATION_DEG['above'][0])
        max_theta = min(150.0, base_theta + INIT_VEL_THETA_DEVIATION_DEG['above'][1])

    v = np.empty(3, dtype=np.float64);
    speed = rng_py.uniform(*INIT_VEL_SPEED_RANGE)
    phi = rng_py.uniform(np.deg2rad(base_phi - INIT_VEL_PHI_DEVIATION_DEG), np.deg2rad(base_phi + INIT_VEL_PHI_DEVIATION_DEG));
    theta = rng_py.uniform(np.deg2rad(min_theta), np.deg2rad(max_theta))
    v[0] = speed * math.sin(theta) * math.cos(phi);
    v[1] = speed * math.sin(theta) * math.sin(phi);
    v[2] = speed * math.cos(theta)
    w = np.zeros(3, dtype=np.float64);
    speed = rng_py.uniform(*INIT_ANG_VEL_SPEED_RANGE)
    phi = rng_py.uniform(0, 2 * math.pi);
    theta = rng_py.uniform(0, math.pi)
    w[0] = speed * math.sin(theta) * math.cos(phi);
    w[1] = speed * math.sin(theta) * math.sin(phi);
    w[2] = speed * math.cos(theta)
    data.qpos[0:3] = r;
    data.qvel[0:3] = v;
    data.qvel[3:6] = w
    return model, data


def find_valid_trajectories_worker(seeds_and_mode):
    seeds, mode, direction = seeds_and_mode
    valid_trajectory_data = []
    for seed in seeds:
        model, data = _init_simulation(seed, mode, direction)
        mujoco.mj_step(model, data)
        positions, velocities, rotations, times, ex_mats, in_mats = [], [], [], [], [], []
        next_save_time = 0.
        while next_save_time < MAX_SIMULATION_TIME:
            steps = round((next_save_time - data.time) / TIMESTEP)
            mujoco.mj_step(model, data, steps)
            correct_side = data.qpos[0] < 0 if direction == 'left_to_right' else data.qpos[0] > 0
            oob_x, oob_y, oob_z_min = OOB_DEFINITIONS[mode]
            is_oob = False

            if mode == 'final_lose':
                # Original code did NOT check which side the ball was on for this mode.
                if abs(data.qpos[0]) > oob_x or abs(data.qpos[1]) > oob_y:
                    is_oob = True
            elif 'final' in mode or 'intermediate' in mode:
                if correct_side and (abs(data.qpos[0]) > oob_x or abs(data.qpos[1]) > oob_y or data.qpos[2] < oob_z_min): is_oob = True
            elif 'first' in mode:
                if mode == 'first_short':
                    if abs(data.qpos[0]) > oob_x or abs(data.qpos[1]) > oob_y or data.qpos[2] < oob_z_min: is_oob = True
                else:  # first_good, first_long
                    if correct_side and (abs(data.qpos[0]) > oob_x or abs(data.qpos[1]) > oob_y): is_oob = True

            if is_oob: break
            ex_mat, in_mat = _calc_cammatrices(data, camera_name=CAMERA_NAME)
            r_cam = world2cam(data.qpos[0:3].copy(), ex_mat)
            r_img = cam2img(r_cam, in_mat[:3, :3])
            if not np.all((r_img >= 0) & (r_img < np.array([WIDTH, HEIGHT]))): break
            positions.append(data.qpos[0:3].copy());
            velocities.append(data.qvel[0:3].copy());
            rotations.append(data.qvel[3:6].copy())
            times.append(next_save_time);
            ex_mats.append(ex_mat);
            in_mats.append(in_mat[:3, :3])
            next_save_time += 1 / FPS
        if len(positions) < MIN_TRAJ_LEN_FRAMES: continue
        hits_opponent, hits_own, hits_ground = _count_hits(positions, direction)
        max_height = MAX_HEIGHT_FIRST_MODE if 'first' in mode else MAX_HEIGHT_OTHER_MODES
        if np.max(np.array(positions)[:, 2]) > max_height: continue
        min_time_for_cut = MIN_TRAJ_CUT_TIME_RATIO * MAX_SIMULATION_TIME
        cut_index = -1
        # (Cutting logic is unchanged from previous correct version)
        if mode == 'final_lose':
            if len(hits_ground) > 0 and hits_ground[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_ground[0], 1, 0)) - 1;
                hits_ground = []
        elif mode == 'final_win':
            if len(hits_opponent) > 2 and hits_opponent[2] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_opponent[2], 1, 0)) - 1;
                hits_opponent = hits_opponent[:2]
            elif len(hits_ground) > 0 and hits_ground[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_ground[0], 1, 0)) - 1
            if cut_index != -1: hits_ground = []
        elif mode == 'intermediate':
            if len(hits_ground) > 0 and hits_ground[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_ground[0], 1, 0)) - 1;
                hits_ground = []
        elif mode == 'first_good':
            if len(hits_opponent) > 1 and hits_opponent[1] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_opponent[1], 1, 0)) - 1;
                hits_opponent = hits_opponent[:1]
            elif len(hits_ground) > 0 and hits_ground[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_ground[0], 1, 0)) - 1
            if cut_index != -1: hits_ground = []
        elif mode == 'first_short':
            if len(hits_own) > 2 and hits_own[2] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_own[2], 1, 0)) - 1;
                hits_own, hits_opponent, hits_ground = hits_own[:2], [], []
            elif len(hits_opponent) > 0 and hits_opponent[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_opponent[0], 1, 0)) - 1;
                hits_opponent, hits_ground = [], []
            elif len(hits_ground) > 0 and hits_ground[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_ground[0], 1, 0)) - 1;
                hits_ground = []
        elif mode == 'first_long':
            if len(hits_ground) > 0 and hits_ground[0] >= min_time_for_cut:
                cut_index = np.sum(np.where(np.array(times) < hits_ground[0], 1, 0)) - 1;
                hits_ground = []
        if cut_index != -1:
            positions, velocities, rotations, times, ex_mats, in_mats = \
                positions[:cut_index], velocities[:cut_index], rotations[:cut_index], times[:cut_index], ex_mats[:cut_index], in_mats[:cut_index]
        if len(positions) < MIN_TRAJ_LEN_FRAMES: continue
        positions_arr = np.array(positions)
        close_to_net_mask = np.abs(positions_arr[:, 0]) < NET_CLEARANCE_X_MARGIN
        if np.any(close_to_net_mask):
            heights_close_to_net = positions_arr[close_to_net_mask, 2];
            widths_close_to_net = positions_arr[close_to_net_mask, 1]
            if np.max(heights_close_to_net) < NET_TOTAL_HEIGHT and np.min(np.abs(widths_close_to_net)) < NET_TOTAL_WIDTH / 2: continue
        is_opposite_site = lambda x: x < 0 if direction == 'left_to_right' else x > 0
        if mode in ['final_lose', 'first_long'] and not is_opposite_site(positions[-1][0]): continue
        valid_bounce_counts = {
            'final_lose': (0, 0, 0), 'final_win': (2, 0, 0), 'intermediate': (1, 0, 0),
            'first_good': (1, 1, 0), 'first_short': (0, 2, 0), 'first_long': (0, 1, 0),
        }
        current_counts = (len(hits_opponent), len(hits_own), len(hits_ground))
        if current_counts == valid_bounce_counts[mode]:
            bounces = sorted(hits_opponent + hits_own)
            valid_trajectory_data.append({
                "positions": np.array(positions), "velocities": np.array(velocities),
                "rotations": np.array(rotations), "times": np.array(times),
                "Mext": np.array(ex_mats), "Mint": np.array(in_mats),
                "bounces": np.array(bounces), "seed": seed
            })
    return valid_trajectory_data


def get_valid_trajectories(num_trajectories, num_processes, mode, direction):
    valid_trajectories = []
    current_seed, batch_size = 0, min(1024, num_trajectories)
    with tqdm.tqdm(total=num_trajectories, unit='trajectory', disable=False) as pbar:
        while len(valid_trajectories) < num_trajectories:
            seeds = [list(range(current_seed + j, current_seed + batch_size, num_processes)) for j in range(num_processes)]
            with Pool(num_processes) as p:
                results_from_pool = p.map(find_valid_trajectories_worker, [(s, mode, direction) for s in seeds if s])
            for trajectory_list in results_from_pool: valid_trajectories.extend(trajectory_list)
            current_seed += batch_size
            pbar.n = len(valid_trajectories)
            pbar.set_postfix_str(f"{len(valid_trajectories)} found");
            pbar.refresh()
    final_trajectories = valid_trajectories[:num_trajectories]
    final_seeds = [traj['seed'] for traj in final_trajectories]
    print(f"Found {len(final_trajectories)} valid solutions.\nSeeds: {final_seeds}")
    return final_trajectories


def save_dataset(path, trajectories_data):
    print(f"Saving {len(trajectories_data)} trajectories to {path}...")
    os.makedirs(path, exist_ok=True)
    for i, traj_data in tqdm.tqdm(enumerate(trajectories_data), total=len(trajectories_data)):
        save_path = os.path.join(path, f"trajectory_{i:04}")
        os.makedirs(save_path, exist_ok=True)
        for key, value in traj_data.items():
            if key != 'seed': np.save(os.path.join(save_path, f'{key}.npy'), value)


def _run_single_simulation(seed, mode, direction):
    worker_args = ([seed], mode, direction)
    found_trajectories = find_valid_trajectories_worker(worker_args)
    if found_trajectories:
        result = found_trajectories[0]
        del result['seed']
        return result
    else:
        return None


if __name__ == "__main__":
    mode = args.mode
    assert mode in OOB_DEFINITIONS, f"Mode {mode} not supported."
    direction = args.direction
    assert direction in ['left_to_right', 'right_to_left'], f"Direction {direction} not supported."

    print(f"Searching for {args.num_trajectories} valid trajectories...")
    all_trajectory_data = get_valid_trajectories(
        num_trajectories=args.num_trajectories,
        num_processes=args.num_processes,
        mode=mode,
        direction=direction
    )

    from paths import data_path

    output_path = os.path.join(data_path, args.folder, mode, direction)
    save_dataset(output_path, all_trajectory_data)

    print("Dataset generation complete.")