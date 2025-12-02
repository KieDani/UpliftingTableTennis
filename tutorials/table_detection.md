# Table Keypoint Detection Tutorial

In this tutorial, you will learn how to detect table tennis table keypoints (corners and net posts) using our `TableDetector`. We will demonstrate how to detect keypoints, filter them using an auxiliary model, and finally use them to calibrate the camera (obtaining intrinsic and extrinsic matrices).

## 1. Installation
First, ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install torch torchvision numpy opencv-python-headless einops tqdm matplotlib scipy scikit-learn omegaconf tomesd pandas tensorboard rich yapf addict
```
Note: You can also use the standard `opencv-python` package.

## 2. Setup and Data Loading
We use torch.hub to easily download the example images provided in the repository and load them into memory. 
You can of course use your custom images instead.

```python
import torch
import cv2
import os
import matplotlib.pyplot as plt

# Define the repository (replace with your actual repo path if different)
repo = 'KieDani/UpliftingTableTennis' 

# 1. Download example images using the helper function defined in hubconf.py
image_folder = torch.hub.load(repo, 'download_example_images', local_folder='example_images', trust_repo=True)

# 2. Load the sequence of images (00.png to 34.png)
images = [cv2.imread(os.path.join(image_folder, f'{i:02d}.png')) for i in range(35)]

print(f"Loaded {len(images)} images.")
```

## 3. Single Model Table Keypoint Detection
Unlike ball detection, our table detection models operate on single frames. The model predicts 13 specific keypoints on the table (corners, net posts, and table center lines).

We will start by loading the primary model, segformerpp_b2.

```python
# 1. Load the primary Table Detection model
# Available models: 'segformerpp_b2', 'segformerpp_b0', 'hrnet'
table_model = torch.hub.load(repo, 'table_detection', model_name='segformerpp_b2', trust_repo=True)

# 2. Run Inference
# The predict function takes a list of single images
print("Running inference with SegFormer++...")
keypoints_main, heatmaps = table_model.predict(images)

# keypoints_main is an array of shape (N, 13, 3) -> [x, y, visibility]
# visibility: 1 = visible, 0 = invisible
print(f"Detected keypoints for {len(keypoints_main)} frames.")
```

## 4. Visualize Raw Detections
Let's visualize the detected keypoints on the first frame.
```python
# Visualize on the first frame
frame_idx = 0
img_vis = images[frame_idx].copy()

# Iterate over the 13 keypoints
for i, (x, y, v) in enumerate(keypoints_main[frame_idx]):
    if v == table_model.KEYPOINT_VISIBLE:
        # Draw green circle
        cv2.circle(img_vis, (int(x), int(y)), 8, (0, 255, 0), -1)
        # Draw keypoint index
        cv2.putText(img_vis, str(i+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

plt.figure(figsize=(12, 8))
plt.title("Detected Table Keypoints (Single Model)")
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

## 5. Auxiliary Model for Keypoint Filtering
Keypoint detections can sometimes jitter slightly between frames or fail due to occlusions (e.g., a player standing in front of the table). 
To improve stability, we use a second auxiliary model (hrnet) and filter the results over the temporal sequence using DBScan.

```python
# 1. Load the auxiliary model
aux_model = torch.hub.load(repo, 'table_detection', model_name='hrnet', trust_repo=True)

# 2. Run Inference with the auxiliary model
print("Running inference with HRNet (Auxiliary)...")
keypoints_aux, _ = aux_model.predict(images)

# 3. Filter the trajectory
# This combines predictions from both models to create a stable result
filtered_keypoints = table_model.filter_trajectory(keypoints_main, keypoints_aux)

print(f"Shape of filtered keypoints: {filtered_keypoints.shape}")
# The result is a single set of stable keypoints (13, 3) representing the table state
```

## 6. Visualize Filtered Keypoints
Let's visualize the filtered keypoints.
```python
# Visualize Filtered Keypoints
# We plot the single set of stable keypoints on the first frame
if frame_idx < len(images):
    img_vis_filtered = images[frame_idx].copy()
    for i, (x, y, v) in enumerate(filtered_keypoints):
        if v == table_model.KEYPOINT_VISIBLE:
            # Plot in blue to distinguish from raw detections
            cv2.circle(img_vis_filtered, (int(x), int(y)), 8, (255, 0, 0), -1) 
            cv2.putText(img_vis_filtered, str(i+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.title("Filtered Table Keypoints (Stable)")
    plt.imshow(cv2.cvtColor(img_vis_filtered, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    print("Displaying filtered keypoints plot...")
    plt.show()
```

## 7. Camera Calibration
Using the stable, filtered keypoints, we can now calibrate the camera. 
This process calculates the Intrinsic Matrix (focal length, optical center) and Extrinsic Matrix (camera position and rotation relative to the table).

Note that our camera calibration algorithm is precise, but not very fast.
If you want to speed up the full pipeline, changing the camera calibration method would be recommended.

```python
# Calculate Camera Matrices
Mint, Mext = table_model.calibrate_camera(filtered_keypoints)

print("\n--- Camera Calibration Results ---")
print("Intrinsic Matrix (K):")
print(Mint)
print("\nExtrinsic Matrix (RT):")
print(Mext)
```


