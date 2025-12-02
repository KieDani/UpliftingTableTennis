from datetime import datetime
import os

from uplifting.helper import get_logs_path as glp

class BaseConfig(object):
    def __init__(self):
        super(BaseConfig, self).__init__()
        self.lr = 1e-4
        self.date_time = datetime.now().strftime("%m%d%Y-%H%M%S")
        self.BATCH_SIZE = 64
        self.NUM_EPOCHS = 800
        self.ident = None
        self.seed = 42
        self.size = None # ['small', 'base', 'large', 'huge']
        self.name = None # ['singlestage', 'multistage']
        self.ema_decay = 0.999
        self.tabletoken_mode = 'stacked' # ['free', 'static', 'dynamic', 'stacked', 'distances']
        self.time_rotation = 'new' # ['old', 'new']

        self.folder = None
        self.exp_id = None # id for describing some (sub-)experiments

    def get_identifier(self):
        if self.ident is None:
            identifier = f'lr:{self.lr:.2e}_bs:{self.BATCH_SIZE:02d}_name:{self.name}_mode:{self.tabletoken_mode}_size:{self.size}_tr:{self.time_rotation}'
            if self.exp_id is not None:
                identifier = identifier + f'_exp:{self.exp_id}'
            identifier = identifier + f'_{self.date_time}'
            self.ident = identifier
        else:
            identifier = self.ident
        return identifier

    def get_logs_path(self, debug=True):
        identifier = self.get_identifier()
        logs_path = os.path.join(glp(), 'logs_tmp') if debug else os.path.join(glp(), 'logs')
        if self.folder is not None:
            logs_path = os.path.join(logs_path, self.folder, identifier)
        else:
            logs_path = os.path.join(logs_path, identifier)
        return logs_path

    def get_pathforsaving(self, debug=True):
        identifier = self.get_identifier()
        if self.folder is not None:
            ident = os.path.join(self.folder, identifier)
        else:
            ident = identifier
        return os.path.join(glp(), 'saved_models_tmp', ident) if debug else os.path.join(glp(), 'saved_models', ident)

    def get_hparams(self):
        hparams = {
            'lr': self.lr,
            'batch_size': self.BATCH_SIZE,
            'num_epochs': self.NUM_EPOCHS,
            'seed': self.seed,
            'size': self.size,
            'name': self.name,
            'ema_decay': self.ema_decay,
            'tabletoken_mode': self.tabletoken_mode,
            'time_rotation': self.time_rotation,
        }
        return hparams


class TrainConfig(BaseConfig):
    def __init__(self, lr, name, size, debug, folder, exp_id=None):
        super(TrainConfig, self).__init__()
        self.lr = lr
        self.size = size
        self.name = name
        self.folder = folder
        self.exp_id = exp_id
        self.debug = debug

        self.randomize_std = 8 # std deviation (in pixels) for randomizing detections
        self.stop_prob = 0.5 # probability to stop the sequence after the bounce
        self.blur_strength = 0.5 # probability to simulate motion blur
        self.randdet_prob = 0.05 # probability to simulate a wrong random detections
        self.randmiss_prob = 0.05 # probability to simulate missing ball detections
        self.tablemiss_prob = 0.05 # probability to simulate missing table detections
        self.transform_mode = 'global' # ['global', 'local']

    def get_hparams(self):
        # add custom hparams
        hparams = super(TrainConfig, self).get_hparams()
        hparams['randomize_std'] = self.randomize_std
        hparams['stop_prob'] = self.stop_prob
        hparams['blur_strength'] = self.blur_strength
        hparams['randdet_prob'] = self.randdet_prob
        hparams['randmiss_prob'] = self.randmiss_prob
        hparams['tablemiss_prob'] = self.tablemiss_prob
        hparams['transform_mode'] = self.transform_mode
        return hparams

    def get_identifier(self):
        identifier = super(TrainConfig, self).get_identifier()
        firstpart = identifier.replace(self.date_time, '')
        identifier = firstpart + f'trans:{self.transform_mode}_' + self.date_time
        return identifier


class EvalConfig(BaseConfig):
    def __init__(self, name, size, tabletoken_mode, transform_mode):
        super(EvalConfig, self).__init__()
        self.name = name
        self.size = size
        self.tabletoken_mode = tabletoken_mode
        self.transform_mode = transform_mode

        self.ident = ''
        self.folder = None

        self.BATCH_SIZE = 128