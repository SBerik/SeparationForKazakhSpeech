import os
import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

from utils import load_config_yml 
from models import get_model
from data import DiarizationDataset

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def main(hparams_file):
    # Loading config file
    cfg_name = Path(hparams_file).stem
    cfg = load_config_yml(hparams_file, cfg_name)
    # Seeds 
    seed_everything(42)
    # Load data 
    datamodule = DiarizationDataset(**cfg['data'])
    # Load model
    model_class = get_model(cfg['xp_config']['model_name'])
    model = model_class(**cfg['model'])
    # TB Log
    logger = TensorBoardLogger('tb_logs', name=cfg_name)
    # Callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', **cfg['early_stop'])
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('checkpoints', cfg_name), **cfg['model_ckpt'])
    # Train
    trainer = pl.Trainer(**cfg['trainer'],
                        logger=logger,
                        enable_checkpointing=checkpoint_callback,
                        callbacks=[checkpoint_callback, early_stop_callback])
    # # Train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/pl_dualpathrnn.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)