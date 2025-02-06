import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter as TensorBoard
from torchmetrics.audio import PermutationInvariantTraining as PIT
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as sisnr

from utils.load_config import load_config 
from utils.training import metadata_info, configure_optimizer
from models import get_model
from trainer import Trainer
from data import DiarizationDataset

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def main(hparams_file):
    # Loading config file    
    cfg = load_config(hparams_file)
    # Load data 
    datamodule = DiarizationDataset(**cfg['data'])
    datamodule.setup(stage = 'fit')
    dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}
    # Load model
    model_class = get_model(cfg['trainer']['model_name'])
    model = model_class(**cfg['model'])
    # Meta-data
    metadata_info(model)
    # TensorBoard
    writer = TensorBoard(f'tb_logs/{Path(hparams_file).stem}', comment = f"{cfg['trainer']['ckpt_folder']}")
    # Optimizer
    optimizer = configure_optimizer(cfg, model)
    # Loss and metrics
    loss_funcs = {name: PIT(func).to(cfg['trainer']['device']) for name, func in {"sisnr": sisnr, "sdr": sdr}.items()}
    # Train
    Trainer(**cfg['trainer']).fit(model, dataloaders, loss_funcs, optimizer, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./configs/sepformer.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)