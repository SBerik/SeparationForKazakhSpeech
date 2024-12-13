import torch
import torchmetrics
from trainer import Trainer 
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter as TensorBoard

from utils.load_config import load_config 
from utils.training import metadata_info, configure_optimizer
from model.model_rnn import Dual_RNN_model
from losses import loss
from dataset import AudioDataModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def main(hparams_file):
    # Loading config file    
    cfg = load_config(hparams_file)
    # Load data 
    datamodule = AudioDataModule(**cfg['data']).setup(stage = 'train')
    dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}
    # Load model
    model = Dual_RNN_model(**cfg['model'])
    # Meta-data
    metadata_info(model)
    # TensorBoard
    writer = TensorBoard(f'tb_logs/{Path(hparams_file).stem}', comment = f"{cfg['trainer']['ckpt_folder']}")
    # Optimizer
    optimizer = configure_optimizer (cfg, model)
    # Train
    Trainer(**cfg['trainer']).fit(model, dataloaders, loss, optimizer, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./config/train_rnn.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)