# Ball Detection Tutorial

# Tutorial: Ball Detection

In this tutorial, you will learn how to detect the table tennis ball in a video sequence using our robust `BallDetector`. 
We will first use a single model to get initial predictions and then improve the results by adding a second auxiliary model for filtering.

## 1. Installation
First, ensure you have the required dependencies installed. You can install them via pip:

```bash
pip install torch torchvision numpy opencv-python einops tqdm matplotlib scipy scikit-learn, omegaconf, tomesd
```

## 2. Setup and Data Loading
We use torch.hub to easily download the example images provided in the repository and load them into memory.
You can of course use your custom images instead.

```python
import torch
import cv2
import os
import matplotlib.pyplot as plt

# Define the repository (replace with your actual repo path if different)
repo = 'kiedani/WACV2026' 

# 1. Download example images using the helper function defined in hubconf.py
# This saves the images to a local folder named 'example_images'
image_folder = torch.hub.load(repo, 'download_example_images', local_folder='example_images', trust_repo=True)

# 2. Load the sequence of images (00.png to 34.png)
# We read them in BGR format (standard for OpenCV)
images = [cv2.imread(os.path.join(image_folder, f'{i:02d}.png')) for i in range(35)]

print(f"Loaded {len(images)} images.")
```

## 3. Single Model Ball Detection
Our ball detection models utilize temporal information. 
Therefore, the input to the model is not just a single frame, but a triplet consisting of the previous, current, and next frame.
We will start by loading the primary model, segformerpp_b2.
```python
# 1. Load the primary Ball Detection model
# Available models: 'segformerpp_b2', 'segformerpp_b0', 'wasb'
ball_model = torch.hub.load(repo, 'ball_detection', model_name='segformerpp_b2', trust_repo=True)

# 2. Prepare the input triplets
# We skip the first and last frame since they don't have a full triplet
image_triples = [(images[i-1], images[i], images[i+1]) for i in range(1, len(images)-1)]

# 3. Run Inference
print("Running inference with SegFormer++...")
ball_positions, heatmaps = ball_model.predict(image_triples)

print(f"Detected {len(ball_positions)} ball positions.")
# ball_positions is an array of shape (N, 3) -> [x, y, confidence]
```

## 4. Visualize Raw Detections
Let's visualize the raw output on one of the frames.
```python
# Visualize the trajectory on a sample frame (e.g., frame 15)
frame_idx = 15
img_vis = images[frame_idx].copy()

# Draw all detected points
for x, y, __ in ball_positions:
    # Draw red circles for raw detections
    cv2.circle(img_vis, (int(x), int(y)), 5, (0, 0, 255), -1) 

plt.figure(figsize=(12, 8))
plt.title("Raw Ball Detections (Single Model)")
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

## 5. Robust Filtering with Auxiliary Model
Single models can sometimes pick up false positives (like white shoe soles or reflections). 
To make the detection robust, we use a second auxiliary model (wasb) to filter the predictions.
This way, we can remove most false-positives.
```python
# 1. Load the auxiliary model
aux_model = torch.hub.load(repo, 'ball_detection', model_name='wasb', trust_repo=True)

# 2. Run Inference with the auxiliary model
print("Running inference with WASB (Auxiliary)...")
aux_positions, _ = aux_model.predict(image_triples)

# 3. Filter the trajectory
# We need to specify the approximate framerate of the video for the physics-based checks
fps = 60.0

filtered_positions, valid_indices, times = ball_model.filter_trajectory(
    ball_positions, 
    aux_positions, 
    fps
)

print(f"Original detections: {len(ball_positions)}")
print(f"Filtered detections: {len(filtered_positions)}")
```

## 6. Visualize Filtered Detections
Now visualize the cleaned trajectory. 
You should see that outliers have been removed, leaving a smooth path.
```python
img_vis_filtered = images[frame_idx].copy()

# Draw filtered points
for x, y in filtered_positions:
    # Draw green circles for filtered detections
    cv2.circle(img_vis_filtered, (int(x), int(y)), 5, (0, 255, 0), -1) 

plt.figure(figsize=(12, 8))
plt.title("Filtered Trajectory (Dual Model)")
plt.imshow(cv2.cvtColor(img_vis_filtered, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```