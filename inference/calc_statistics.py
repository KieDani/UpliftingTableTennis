'''Calculate statistics for the TTHQ, TTST, and Blurball datasets.'''
import os
import numpy as np
import torch
import scipy

from balldetection.dataset import TTHQ as TTHQ_ball, BlurBall as BlurBall_ball
from tabledetection.dataset import TTHQ as TTHQ_table, BlurBall as BlurBall_table
from inference.dataset import TTHQ as TTHQ_trajectory, TTST as TTST_trajectory
from uplifting.data import RealInferenceDataset as TTST_synthetic, TableTennisDataset as SimulatedDataset



def calc_statistics():
    # Number of frames with ball annotations
    modes = ['train', 'val', 'test']
    for mode in modes:
        dataset = TTHQ_ball(mode=mode, heatmap_sigma=6, in_frames=3, use_invisible=False)
        num_frames = len(dataset)
        print('\n----------------------------')
        print(f'TTHQ {mode} ball num frames: {num_frames}')
        print('----------------------------\n')
    for mode in modes:
        dataset = BlurBall_ball(mode=mode, heatmap_sigma=6, in_frames=3, use_invisible=False)
        num_frames = len(dataset)
        print('\n----------------------------')
        print(f'Blurball {mode} ball num frames: {num_frames}')
        print('----------------------------\n')

    # Number of frames with table annotations
    for mode in modes:
        dataset = TTHQ_table(mode=mode, heatmap_sigma=6)
        num_frames = len(dataset)
        print('\n----------------------------')
        print(f'TTHQ {mode} table num frames: {num_frames}')
        print('----------------------------\n')
    for mode in modes:
        dataset = BlurBall_table(mode=mode, heatmap_sigma=6)
        num_frames = len(dataset)
        print('\n----------------------------')
        print(f'Blurball {mode} table num frames: {num_frames}')
        print('----------------------------\n')

    # Number of trajectories
    mode = 'test'
    dataset = TTHQ_trajectory()
    num_trajectories = len(dataset)
    print('\n----------------------------')
    print(f'TTHQ {mode} trajectory num trajectories: {num_trajectories}')
    print('----------------------------\n')
    dataset = TTST_trajectory()
    num_trajectories = len(dataset)
    print('\n----------------------------')
    print(f'TTST {mode} trajectory num trajectories: {num_trajectories}')
    print('----------------------------\n')
    for mode in modes:
        dataset = SimulatedDataset(mode=mode)
        num_trajectories = len(dataset)
        print('\n----------------------------')
        print(f'Simulated {mode} trajectory num trajectories: {num_trajectories}')
        print('----------------------------\n')

    # Number of ball positions in TTST
    for mode in ['val', 'test']:
        dataset = TTST_synthetic(mode=mode)
        num_pos = 0
        for stuff in dataset:
            mask = stuff[2]
            num_pos += int(mask.sum().item())
        print('\n----------------------------')
        print(f'TTST {mode} num ball positions: {num_pos}')
        print('----------------------------\n')

    # Number of ball positions in Simulated
    for mode in modes:
        dataset = SimulatedDataset(mode=mode)
        num_pos = 0
        for stuff in dataset:
            mask = stuff[2]
            num_pos += int(mask.sum().item())
        print('\n----------------------------')
        print(f'Simulated {mode} num ball positions: {num_pos}')
        print('----------------------------\n')



if __name__ == '__main__':
    calc_statistics()
