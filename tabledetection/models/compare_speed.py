import torch
from tqdm import tqdm
from time import time

from tabledetection.train import get_model
from tabledetection.config import TrainConfig

device = 'cuda:0'

def compare_speed():
    B = 8
    with torch.no_grad():
        for model_name in ['segformerpp_b0', 'segformerpp_b2', 'hrnet', 'vitpose']:
            print(f'Benchmarking model: {model_name}')
            fake_config = TrainConfig(lr=1, model_name=model_name, heatmap_sigma=6, pretraining=True, dataset_name='tthq', exp_id='', folder='', debug=False)
            model = get_model(model_name, resolution=fake_config.image_resolution, pretraining=False).to(device)
            start_time = time()
            epochs = 50
            for i in tqdm(range(epochs)):
                random_input = torch.randn((B, 3, fake_config.image_resolution[1], fake_config.image_resolution[0]), device=device)
                _ = model(random_input)
            end_time = time()
            avg_time = (end_time - start_time) / epochs
            print(f'Average inference time per batch of size {B}: {avg_time:.4f} seconds')
            print(f'FPS: {B / avg_time:.2f}')
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'Number of parameters: {num_params / 1e6:.2f}M')
            resolution = fake_config.image_resolution
            print(f'Resolution: {resolution[0]}x{resolution[1]}')
            print('---')
            del model



if __name__ == '__main__':
    compare_speed()




