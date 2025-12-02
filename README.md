# Official Implementation of "Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation"
___
We are happy to announce that our paper "Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation" has been accepted at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026!

[Paper Link](https://arxiv.org/abs/2511.20250) | [Project Page](https://kiedani.github.io/WACV2026/index.html)



https://github.com/user-attachments/assets/58717aed-dc5f-4193-b269-40102e0f81a2



# Installation

### Inference Only
Use this installation if you only want to use our final models, but do not plan to simulate the synthetic dataset or do some ablation studies.
Tested with python version 3.12.4.
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```


### Full Installation
Use this installation if you want to do everything, including training, testing and data simulation.
Tested with python version 3.12.4. 

VitPose depends on mmcv, which sometimes is tricky to install. 
Make sure you don't use a conda virtual environment, instead use a pip venv.
If you don't use VitPose, you can skip mmcv and mmcv-full installation.
However, make sure to uncomment the VitPose model_path in the inference_balldetection.py and inference_tabledetection.py scripts.

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124

pip install -r requirements_full.txt

pip install mmcv==2.2.0
```

# Easy usage of our models
In this section, we explain how to load our trained models via torch.hub.
You can either load our full pipeline, or use the individual models separately.
### Full Pipeline
You find a tutorial on how to use the full pipeline easily via torch.hub [here](tutorials/full_pipeline.md).
### Ball Detection
If you simply want to use our ball detection models, you can find a tutorial [here](tutorials/ball_detection.md).
### Table Keypoint Detection and Camera Calibration
If you are interested in detecting the table keypoints and calibrating the camera, you can find a tutorial [here](tutorials/table_detection.md).


# Reproduce Results
In the file ``paths.py``, the important paths have to be defined.

## Datasets
The paths to the datasets should be defined in the ``data_path`` variable in ``paths.py``.
You can choose an arbitrary folder on your system.
```python
# paths.py
data_path = '<path_to_your_data_folder>'
```
#### TTHQ
We created the TTHQ dataset for this paper.
To obtain the videos, have a look at ``video_list.txt``.
There, we provide some suggestions how to download the videos and explain how to process them using ffmpeg.
Create the folder ``tthq_videos`` in the ``data_path`` directory and copy the downloaded videos there.
Download ``tthq_annotation.zip`` from [here](https://mediastore.rz.uni-augsburg.de/get/E6idNDRk20/) and unzip it in the folder ``tthq_videos``.
Finally, run the script ``extract_tthq_dataset.py`` to create the TTHQ dataset:
```bash
python -m dataprocessing.extract_tthq_dataset
```
Note: The data processing takes a while.

#### TTST
The TTST dataset is introduced in the paper [Towards ball spin and trajectory analysis in table tennis broadcast videos via physically grounded synthetic-to-real transfer. CVPRW 2025](https://kiedani.github.io/CVPRW2025/).
We corrected a little mistake in the data processing. 
You can download the updated dataset [here](https://mediastore.rz.uni-augsburg.de/get/jprwueaZYd/).
Simply unpack the downloaded file in the ``data_path`` directory.

> Note: This dataset does not include the frames, so you can use it to evaluate the uplifting model, but not for the full pipeline. 
> To get the full dataset, please contact us.


#### Blurball
The blurball dataset is introduced in the paper [BlurBall: Joint Ball and Motion Blur Estimation for Table Tennis Ball Tracking](https://cogsys-tuebingen.github.io/blurball/).
We use it for the pretraining of our ball detection models.
In the ``data_path`` directory, create a folder ``blurball`` and download the dataset from [here](https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7).
Follow the instructions in the downloaded README to properly prepare the dataset.


#### Synthetic Dataset
For the training of the 2D-to-3D uplifting model, we created a synthetic (but physically accurate) dataset using MuJoCo.
To create the dataset, you have to run the script ``mujocosimulation.py``. First,
```bash
xvfb-run -a -s "-screen 0 1400x900x24" bash

python -m syntheticdataset.mujocosimulation --num_trajectories 50000 --num_processes 96 --mode intermediate --direction left_to_right --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 50000 --num_processes 96 --mode intermediate --direction right_to_left --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode first_good --direction left_to_right --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode first_good --direction right_to_left --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_short --direction left_to_right --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_short --direction right_to_left --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_long --direction left_to_right --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 2500 --num_processes 96 --mode first_long --direction right_to_left --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_win --direction left_to_right --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_win --direction right_to_left --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_lose --direction left_to_right --folder syntheticdata 
python -m syntheticdataset.mujocosimulation --num_trajectories 5000 --num_processes 96 --mode final_lose --direction right_to_left --folder syntheticdata
```
This takes a significant amount of time. 
You can speed it up by using more processes (if your system has enough CPU cores), simply adjust the ``--num_processes`` argument.

Because this process can take several days, we also provide the synhetic dataset for download. It is split into 3 parts, you can download them from:
- [part 1](https://mediastore.rz.uni-augsburg.de/get/zj3aBN4U9N/ - 1.5GB)
- [part 2](https://mediastore.rz.uni-augsburg.de/get/gCq5s6s8EJ/  - 1.5GB)
- [part 3](https://mediastore.rz.uni-augsburg.de/get/9HIr7_yQFe/   - 1.0GB)

Save the downloaded files in the ``data_path`` directory.
Then, recombine them and unzip them:
```bash
cat syntheticdata.zip.part* > syntheticdata.zip
unzip syntheticdata.zip
rm syntheticdata.zip.part*
```



## Weights
Download the weights from [here](https://mediastore.rz.uni-augsburg.de/get/TL7oQRStHG/) and unzip them. 
The path to the unzipped folder should be set in the ``weights_path`` variable in ``paths.py``:
```python
# paths.py
weights_path = '<path_to_unzipped_folder(_with_the_name_weights)>'
```


## Inference
In this section we describe how to run the evaluations in the paper.

#### Ball Detection
To evaluate the ball detection, run the script ``inference_balldetection.py``:
```bash
python -m inference.inference_balldetection --gpu 0
```
If your system has multiple GPUs, you can specify which one is used via the ``--gpu`` argument.
> Note: If you did not install mmcv, please comment the VitPose model path in the ``model_paths`` list in ``inference_balldetection.py``

#### Table Keypoint Detection
To evaluate the table keypoint detection, run the script ``inference_tabledetection.py``:
```bash
python -m inference.inference_tabledetection --gpu 0
```
If your system has multiple GPUs, you can specify which one is used via the ``--gpu`` argument.
> Note: If you did not install mmcv, please comment the VitPose model path in the ``model_paths`` list in ``inference_tabledetection.py``

#### 2D-to-3D Uplifting
To evaluate the 2D-to-3D uplifting, run the script ``inference_uplifting.py``:
```bash
python -m inference.inference_uplifting --gpu 0
```
If your system has multiple GPUs, you can specify which one is used via the ``--gpu`` argument.

#### Full Pipeline
To evaluate the full pipeline, run the script ``inference_combined.py``:
```bash
python -m inference.inference_combined --dataset tthq --gpu 0 
```
If your system has multiple GPUs, you can specify which one is used via the ``--gpu`` argument.
If you requested the full dataset from us, you can set ``--dataset ttst`` to evaluate on the TTST dataset.
> Note: If you are interested in our camera calibration algorithm, have a look at the inference_combined.py script to see the usage.


## Training
For performing training, you have to set ``logs_path`` in ``paths.py`` to a folder where the training logs and model checkpoints should be saved:
```python
# paths.py
logs_path = '<path_to_your_logs_folder>'
```
You can choose an arbitrary folder on your system.

#### Ball Detection
To train the ball detection models, run the script ``balldetection/train.py``:
```bash
python -m balldetection.train--gpu 0 --folder results --pretraining --model <model_name>
```
Replace ``<model_name>`` with one of the following options: ``segformerpp_b0``, ``segformerpp_b2``, ``wasb``, ``vitpose``.
The ``--pretraining`` argument indicates that the weight from the pretraining on the blurball dataset are used for initialization. 
We already provide these weights, so you don't have to do the pretraining yourself.

If you want to do the pretraining on the blurball dataset yourself, run:
```bash
python -m balldetection.train --gpu 0 --folder pretraining --model <model_name> --dataset blurball
```
If your system has multiple GPUs, you can specify which one is used via the ``--gpu`` argument.

#### Table Keypoint Detection
To train the table keypoint detection models, run the script ``tabledetection/train.py``:
```bash
python -m tabledetection.train --gpu 0 --folder results --model <model_name>
```
Replace ``<model_name>`` with one of the following options: ``segformerpp_b0``, ``segformerpp_b2``, ``hrnet``, ``vitpose``.
Note that we do not use any pretraining for the table keypoint detection.
If your system has multiple GPUs, you can specify which one is used via the ``--gpu`` argument.

#### 2D-to-3D Uplifting
To train the 2D-to-3D uplifting model, run the script ``uplifting/train.py``:
```bash
python -m uplifting.train --gpu 0 --folder results --token_mode <token_mode> --time_rotation <time_rotation>
```
Replace ``<token_mode>`` and ``<time_rotation>`` with one of the following options to reproduce the results in the paper:
- Kienzle et al: ``--token_mode originalmethod --time_rotation old``
- Mixed: ``--token_mode originalmethod --time_rotation new``
- Ours: ``--token_mode dynamic --time_rotation new``


# Citation
If you find our work useful in your research, please consider citing our paper:
```
@inproceedings{kienzle2026uplifting,
  title={Uplifting Table Tennis: A Robust, Real-World Application for 3D Trajectory and Spin Estimation},
  author={Kienzle, Daniel and Ludwig, Katja and Lorenz, Julian and Satoh, {Shin'ichi} and Lienhart, Rainer},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```
