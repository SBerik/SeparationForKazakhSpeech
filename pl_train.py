import os
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.audio import PermutationInvariantTraining as PIT
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as sisnr

from utils.load_config import load_config_yml 
from models import get_model
from data import DiarizationDataset

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def main(hparams_file):
    # Loading config file    
    cfg = load_config_yml(hparams_file)
    # Load data 
    datamodule = DiarizationDataset(**cfg['data'])
    # Load model
    model_class = get_model(cfg['xp_config']['model_name'])
    model = model_class(**cfg['model'])
    # TB Log
    os.makedirs(f'tb_logs/{Path(hparams_file).stem}', exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(f'tb_logs/{Path(hparams_file).stem}', **cfg['tb_logger'])
    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", **cfg['early_stop'])
    os.makedirs(f'checkpoints/{Path(hparams_file).stem}', exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/{Path(hparams_file).stem}', **cfg['model_ckpt'])
    # Train
    trainer = pl.Trainer(**cfg['trainer'],
                        logger=logger,
                        enable_checkpointing=checkpoint_callback,
                        callbacks=[checkpoint_callback, early_stop_callback])
    # Train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/pl_dualpathrnn.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)