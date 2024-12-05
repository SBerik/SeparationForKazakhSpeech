import torch
import torchmetrics
from trainer import Trainer 
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter as TensorBoard

from utils.load_config import load_config 
from utils.training import metadata_info
# from models import VADNet 
# from dataset import *

from dataset import make_dataloader
from model.model_rnn import Dual_RNN_model
from losses import loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

def main(hparams_file):
    # Loading config file    
    cfg, ckpt_folder = load_config(hparams_file)
    # Load data 
    # train_dataloader, val_dataloader = make_dataloader(**cfg['data'])
    # dataloaders = {'train': train_dataloader, 'valid': val_dataloader}
    # Load model
    model = Dual_RNN_model(**cfg['model'])
    # Meta-data
    metadata_info(model)
    # TensorBoard
    writer = TensorBoard(f'tb_logs/{Path(hparams_file).stem}', comment = f'{ckpt_folder}')
    # Optimizer
    assert cfg['training']["optim"] in ['Adam', 'SGD'], "Invalid optimizer type"
    optimizer = (torch.optim.Adam if cfg['training']["optim"] == 'Adam' else torch.optim.SGD) (model.parameters(), 
                 lr=cfg['training']["lr"], weight_decay=cfg['training']["weight_decay"])
    return
    # Train
    Trainer(**cfg['trainer'], ckpt_folder = ckpt_folder).fit(model, dataloaders, loss, optimizer, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, default="./config/train_rnn.yml", help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)