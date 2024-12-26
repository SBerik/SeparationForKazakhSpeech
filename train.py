import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter as TensorBoard
from torchmetrics.audio import SignalDistortionRatio

from losses import sisnr_loss, sdr_loss
from utils.load_config import load_config 
from utils.training import metadata_info, configure_optimizer
from models import MODELS
from trainer import Trainer
from data.DiarizationDataset import DiarizationDataset

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def main(hparams_file):
    # Loading config file    
    cfg = load_config(hparams_file)
    # Load data 
    datamodule = DiarizationDataset(**cfg['data']).setup(stage = 'train')
    dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}
    # Load model
    model_class  = MODELS(**cfg['xp_config']['model_type'])
    model = model_class(**cfg['model'])
    # Meta-data
    metadata_info(model)
    # TensorBoard
    writer = TensorBoard(f'tb_logs/{Path(hparams_file).stem}', comment = f"{cfg['trainer']['ckpt_folder']}")
    # Optimizer
    optimizer = configure_optimizer (cfg, model)
    # Train
    Trainer(**cfg['trainer']).fit(model, dataloaders, sisnr_loss, optimizer, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/train_rnn.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)