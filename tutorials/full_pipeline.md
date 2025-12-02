# End-to-End Uplifting Pipeline Tutorial

This tutorial combines ball detection, table detection, and our novel uplifting model to estimate the 3D trajectory and spin of the ball from a monocular video.

## 1. Installation
First, ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install torch torchvision numpy opencv-python einops tqdm matplotlib scipy scikit-learn omegaconf tomesd
```

## 2. Setup and Data Loading
We use torch.hub to easily download the example images provided in the repository and load them into memory. 
You can of course use your custom images instead.
```python
import torch
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the repository (replace with your actual repo path if different)
repo = 'kiedani/WACV2026' 

# 1. Download example images using the helper function defined in hubconf.py
image_folder = torch.hub.load(repo, 'download_example_images', local_folder='example_images', trust_repo=True)

# 2. Load the sequence of images (00.png to 34.png)
images = [cv2.imread(os.path.join(image_folder, f'{i:02d}.png')) for i in range(35)]
fps = 60.0  # The framerate is required for physics calculations

print(f"Loaded {len(images)} images.")
```

## 3. Run the Full Pipeline
We will sequentially run ball detection, table detection, and then the uplifting model to get the 3D trajectory and spin.
- Ball Detection: Using the segformerpp_b2 model. Additionally uses an auxiliary wasb model for filtering.
- Table Detection: Using the segformerpp_b2 model. Additionally uses an auxiliary hrnet model for filtering.
- Uplifting: Combines the 2D ball positions and table keypoints to estimate 3D trajectory and spin.
```python
# 1. Load the Full Pipeline
# This downloads and loads all necessary weights automatically
pipeline = torch.hub.load(repo, 'full_pipeline', trust_repo=True)

# 2. Run Inference
# The pipeline handles ball detection, table detection, filtering, 
# normalization, and 3D uplifting internally.
print("Running full pipeline (this may take a moment)...")
pred_spin, pred_pos_3d = pipeline.predict(images, fps)

print(f"Predicted 3D positions shape: {pred_pos_3d.shape}")
print(f"Predicted Spin Vector (local coords): {pred_spin}")
```

## Analyzing the Results
We can now interpret the predicted spin and verify the 3D trajectory by reprojecting it back onto the 2D image.

```python
# Identify Spin Type
# In our local coordinate system, the y-axis corresponds to the top-backspin axis.
spin_magnitude = pred_spin[1] / (2 * np.pi)  # Convert rad/s to Hz
if spin_magnitude > 2:
    spin_type = "Topspin"
elif spin_magnitude < -2:
    spin_type = "Backspin"
else:
    spin_type = "No significant spin"

print(f"Predicted Spin Class: {spin_type}")
print(f"Spin Magnitude: {spin_magnitude:.1f} Hz")

# 2. Reproject 3D points to 2D for visualization
# We need the camera matrices to project 3D world points -> 2D image pixels.
# The pipeline can calibrate the camera using the detected table keypoints.

# (Note: We use the detections from the first image for calibration here)
table_det_model = torch.hub.load(repo, 'table_detection', model_name='segformerpp_b2', trust_repo=True)
kps, _ = table_det_model.predict(images)
# We filter/smooth these internally in the pipeline, but for visualization 
# let's just use the raw keypoints from the first frame to get matrices.
Mint, Mext = pipeline.calibrate_camera(kps[0]) 

reprojected_2d = pipeline.reproject(pred_pos_3d, Mint, Mext)
```

## 5. Visualization
Finally, let's plot the result. We will draw the reprojected 3D points. 
If the uplifting model works correctly, these points should align perfectly with the ball in the video, confirming that the 3D estimation is accurate.
```python
# Plot on a sample frame
frame_idx = 15
img_vis = images[frame_idx].copy()

# Draw reprojected 3D points in Cyan
for x, y in reprojected_2d:
    cv2.circle(img_vis, (int(x), int(y)), 5, (255, 255, 0), -1)

plt.figure(figsize=(14, 8))
plt.title(f"3D Uplifted Trajectory Reprojected (Spin: {spin_type})")
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```