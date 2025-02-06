import yaml
import os 
from pathlib import Path

def load_config (hparams_file):
    cfg = yaml.load(open(hparams_file), Loader=yaml.FullLoader)
    ckpt_folder = os.path.join('./checkpoints', Path(hparams_file).stem).replace('\\', '/')
    os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, 'hparams.yml'), 'w') as file:
        yaml.dump(cfg, file)
    cfg['trainer']['ckpt_folder'] = ckpt_folder
    return cfg

def load_config_yml(hparams_file, cfg_name):
    cfg = yaml.load(open(hparams_file), Loader=yaml.FullLoader)
    ckpt_folder = os.path.join('./checkpoints', cfg_name).replace('\\', '/')
    os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, 'hparams.yml'), 'w') as file:
        yaml.dump(cfg, file)
    return cfg