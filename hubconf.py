dependencies = ['torch', 'numpy', 'cv2', 'einops', 'tqdm', 'sklearn', 'matplotlib', 'pandas', 'scipy', 'omegaconf', 'tomesd']

import torch
import os
from interface import BallDetector, TableDetector, TableTennisPipeline, _get_weights_path

REPO_BASE_URL = "https://github.com/KieDani/UpliftingTableTennis/tree/main/"

def ball_detection(model_name='segformerpp_b2', **kwargs):
    """
    Loads the Ball Detection Model.
    Available models: 'segformerpp_b2', 'segformerpp_b0', 'wasb'
    """
    return BallDetector(model_name=model_name)


def table_detection(model_name='segformerpp_b2', **kwargs):
    """
    Loads the Table Detection Model.
    Available models: 'segformerpp_b2', 'segformerpp_b0'
    """
    return TableDetector(model_name=model_name)


def full_pipeline():
    """
    Loads the End-to-End Pipeline (Ball + Table + Uplifting).
    """
    return TableTennisPipeline()


def download_example_images(local_folder='example_images'):
    """
    Downloads the example images from the official repository to a local folder.
    """
    os.makedirs(local_folder, exist_ok=True)
    print(f"Downloading example images to '{local_folder}'...")

    # We assume the tutorial has 35 images named 00.png to 34.png
    for i in range(35):
        filename = f"{i:02d}.png"
        url = f"{REPO_BASE_URL}/tutorials/example_imgs/{filename}"
        save_path = os.path.join(local_folder, filename)

        if not os.path.exists(save_path):
            try:
                torch.hub.download_url_to_file(url, save_path, progress=False)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")

    print("Download complete.")
    return local_folder
