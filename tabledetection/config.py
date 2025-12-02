from datetime import datetime
import os

from tabledetection.helper_tabledetection import get_logs_path as glp


class TrainConfig(object):
    def __init__(self, lr, model_name, heatmap_sigma, pretraining, dataset_name, exp_id, folder, debug):
        '''
        Config class for training the model.
        Args:
            lr (float): Learning rate.
            model_name (str): Name of the model.
            heatmap_sigma (float): Sigma for Gaussian heatmap generation.
            pretraining (bool): Flag for model pretraining.
            folder (str): Folder name for saving logs and models.
            debug (bool): Debug mode flag.
        '''
        self.lr = lr
        self.model_name = model_name
        self.debug = debug
        self.heatmap_sigma = heatmap_sigma
        self.folder = folder
        self.dataset_name = dataset_name
        self.exp_id = exp_id if exp_id is not None else 'None'

        self.image_resolution = self._get_image_resolution()
        self.model_pretraining = pretraining
        self.BATCH_SIZE = 4
        self.NUM_EPOCHS = 700
        if dataset_name == 'tabletennis':
            self.VAL_ITERATIONS = 16
        elif dataset_name == 'tthq':
            self.VAL_ITERATIONS = 32
        elif dataset_name == 'blurball':
            self.VAL_ITERATIONS = 2000
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Please specify a valid dataset name.")
        self.seed = 42
        self.ema_alpha = 0.999
        self.date_time = datetime.now().strftime("%m%d%Y-%H%M%S")
        self.ident = f'mn:{self.model_name}-hs:{self.heatmap_sigma:.2f}-lr:{self.lr:.2e}-dn:{self.dataset_name}-mp:{self.model_pretraining}-' \
                     f'bs:{self.BATCH_SIZE:02d}-exp:{self.exp_id}-{self.date_time}'

        self.logs_path = os.path.join(glp(), 'logs_tmp' if self.debug else 'logs', self.folder, self.ident)
        self.saved_models_path = os.path.join(glp(), 'saved_models' if not self.debug else 'saved_models_tmp', self.folder, self.ident)

    def get_hparams(self):
        hparams = {
            'lr': self.lr,
            'batch_size': self.BATCH_SIZE,
            'num_epochs': self.NUM_EPOCHS,
            'seed': self.seed,
            'model_name': self.model_name,
            'debug': self.debug,
            'logs_path': self.logs_path,
            'saved_models_path': self.saved_models_path,
            'date_time': self.date_time,
            'ident': self.ident,
            'heatmap_sigma': self.heatmap_sigma,
            'ema_alpha': self.ema_alpha,
            'image_resolution': self.image_resolution,
            'model_pretraining': self.model_pretraining,
            'dataset_name': self.dataset_name,
            'exp_id': self.exp_id,
        }
        return hparams

    def _get_image_resolution(self):
        if self.model_name in ['segformerpp_b0']:
            return (1920, 1088)
        elif self.model_name in ['segformerpp_b1', 'segformerpp_b2']:
            return (1600, 896)
        elif self.model_name in ['segformerpp_b3', 'segformer_b4', 'segformer_b5']:
            return (1280, 704)
        elif self.model_name in ['vitpose']:
            return (1152, 640)
        elif self.model_name in ['hrnet']:
            return (1280, 704)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}. Please specify a valid model name.")
