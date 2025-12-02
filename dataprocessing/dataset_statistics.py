'''This file can be used to calculate some camera statistics -> Results are interesting for choosing parameters for the syntheticdataset'''
import os
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
import ast
import matplotlib.pyplot as plt

from paths import data_path as DATA_PATH
from syntheticdataset.helper import get_cameralocations


data_path = os.path.join(DATA_PATH, 'tthq')

def camera_statistics():
    load_path = os.path.join(data_path, 'camera_matrices.csv')
    df = pd.read_csv(load_path, sep=';')

    data_per_video = {}
    # load camera matrices per video
    for video, M_int, M_ext in zip(df['video'], df['M_int'], df['M_ext']):
        M_int = np.array(ast.literal_eval(M_int), dtype=np.float32)
        M_ext = np.array(ast.literal_eval(M_ext), dtype=np.float32)
        video = int(video)

        if video not in data_per_video.keys(): data_per_video[video] = []
        data_per_video[video].append((M_int, M_ext))

    # compute statistics for each video -> fx, fy, camera distance, phi
    stats_per_video = {}
    overall_stats = []
    for video in tqdm(data_per_video.keys()):
        stats_per_video[video] = []
        for M_int, M_ext in data_per_video[video]:
            fx = np.abs(M_int[0, 0])
            fy = np.abs(M_int[1, 1])
            camera_position = get_cameralocations(M_ext)
            camera_distance = np.linalg.norm(camera_position)
            phi = np.arctan2(camera_position[1], camera_position[0]) * 180 / np.pi
            stats_per_video[video].append((fx, fy, camera_distance, phi))
            overall_stats.append((fx, fy, camera_distance, phi))

    stats_per_video = {k: np.array(v) for k, v in stats_per_video.items()}
    overall_stats = np.array(overall_stats)

    # evalue and print statistics -> mean, median, std, max, min
    for video in stats_per_video.keys():
        stats = stats_per_video[video]
        print(f'Video {video}:')
        print(f'  fx: mean={np.mean(stats[:, 0]):.2f}, median={np.median(stats[:, 0]):.2f}, std={np.std(stats[:, 0]):.2f}, max={np.max(stats[:, 0]):.2f}, min={np.min(stats[:, 0]):.2f}')
        print(f'  fy: mean={np.mean(stats[:, 1]):.2f}, median={np.median(stats[:, 1]):.2f}, std={np.std(stats[:, 1]):.2f}, max={np.max(stats[:, 1]):.2f}, min={np.min(stats[:, 1]):.2f}')
        print(f'  camera distance: mean={np.mean(stats[:, 2]):.2f}, median={np.median(stats[:, 2]):.2f}, std={np.std(stats[:, 2]):.2f}, max={np.max(stats[:, 2]):.2f}, min={np.min(stats[:, 2]):.2f}')
        print(f'  phi: mean={np.mean(stats[:, 3]):.2f}, median={np.median(stats[:, 3]):.2f}, std={np.std(stats[:, 3]):.2f}, max={np.max(stats[:, 3]):.2f}, min={np.min(stats[:, 3]):.2f}')
        print()
    print(f'Overall statistics:')
    print(f'  fx: mean={np.mean(overall_stats[:, 0]):.2f}, median={np.median(overall_stats[:, 0]):.2f}, std={np.std(overall_stats[:, 0]):.2f}, max={np.max(overall_stats[:, 0]):.2f}, min={np.min(overall_stats[:, 0]):.2f}')
    print(f'  fy: mean={np.mean(overall_stats[:, 1]):.2f}, median={np.median(overall_stats[:, 1]):.2f}, std={np.std(overall_stats[:, 1]):.2f}, max={np.max(overall_stats[:, 1]):.2f}, min={np.min(overall_stats[:, 1]):.2f}')
    print(f'  camera distance: mean={np.mean(overall_stats[:, 2]):.2f}, median={np.median(overall_stats[:, 2]):.2f}, std={np.std(overall_stats[:, 2]):.2f}, max={np.max(overall_stats[:, 2]):.2f}, min={np.min(overall_stats[:, 2]):.2f}')
    print(f'  phi: mean={np.mean(overall_stats[:, 3]):.2f}, median={np.median(overall_stats[:, 3]):.2f}, std={np.std(overall_stats[:, 3]):.2f}, max={np.max(overall_stats[:, 3]):.2f}, min={np.min(overall_stats[:, 3]):.2f}')


    # histograms
    labels = ['fx', 'fy', 'camera distance', 'phi']
    plt.figure(figsize=(12, 8))
    for i in enumerate(labels):
        plt.subplot(2, 2, i[0]+1)
        plt.hist(overall_stats[:, i[0]], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram of {i[1]}')
        plt.xlabel(i[1])
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    camera_statistics()