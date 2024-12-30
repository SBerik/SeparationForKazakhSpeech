import os
import argparse

import torch 

from models import MODELS
from utils.load_config import load_config 
from utils.data_processing import *

class Separation:
    def __init__(self, cfg, weight, device = 'cpu'):
        self.device = device
        model_class = MODELS[cfg['xp_config']['model_type']]
        self.net = model_class(**cfg['model'])
        self.net.to(device)
        dicts = torch.load(weight, map_location=device, weights_only=False)
        self.net.load_state_dict(dicts['model_state_dict'])
    
    def separate(self, mixed_sample, file_path):
        name = get_file_name(mixed_sample)
        mixed = read_wav(mixed_sample)
        self.net.eval()
        with torch.no_grad():
            norm = torch.norm(mixed, float('inf'))
            mixed = torch.unsqueeze(mixed, 0).to(self.device)
            ests = self.net(mixed)
            spks = [torch.squeeze(s.detach()) for s in ests]
            for index, s in enumerate(spks):
                s = s - torch.mean(s)
                s = s * norm / torch.max(torch.abs(s))
                os.makedirs(file_path + '/spk' + str(index+1), exist_ok=True)
                filename = file_path + '/spk' + str(index+1) + '/inferenced_' + name + '.flac'
                write_wav(filename, s.unsqueeze(0).cpu(), 16000)
                print('saved in:', filename)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--mixed_sample', type=str, default='./samples/5ed8a9a769a27_5f885557c7f91.flac', help='Path to mix scp file.')
    parser.add_argument('-c', '--config', type=str, default='./configs/train_rnn.yml', help='Path to yaml file.')
    parser.add_argument('-w', '--weight', type=str, default='./weights/DualPath_RNN_179_-3.1895.pt', help="Path to model file.")
    parser.add_argument('-s', '--save_path', type=str, default='./samples', help='save result path')
    args = parser.parse_args()
    cfg = load_config(args.config)
    separator = Separation(cfg, args.weight, device='cuda')
    separator.separate(args.mixed_sample, args.save_path)
