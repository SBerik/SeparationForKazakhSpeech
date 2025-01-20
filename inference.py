import os
import argparse

import torch 

from models import MODELS
from utils.load_config import load_config 
from utils.data_processing import *


class Separation:
    def __init__(self, cfg_path, weight_path, device = 'cpu'):
        cfg = load_config(cfg_path)
        self.model_type = cfg['xp_config']['model_type']
        model_class = MODELS[self.model_type]
        self.net = model_class(**cfg['model'])
        self.device = device
        self.net.to(device)
        dicts = torch.load(weight_path, map_location=device, weights_only=False)
        self.net.load_state_dict(dicts['model_state_dict'])
    
    def predict(self, mixed):
        mixed = torch.unsqueeze(mixed, 0).to(self.device)  
        return self.net(mixed) 

    def separate(self, mixed_sample, file_path):
        name = get_file_name(mixed_sample)
        mixed = read_wav(mixed_sample)
        self.net.eval()
        with torch.no_grad():
            norm = torch.norm(mixed, float('inf'))
            ests = self.predict(mixed)
            spks = [torch.squeeze(s.detach()) for s in ests]
            files_name = name.split('_')[::-1]
            for index, s in enumerate(spks):
                s = s - torch.mean(s)
                s = s * norm / torch.max(torch.abs(s))
                os.makedirs(file_path + '/' + self.model_type + '/spk' + str(index+1), exist_ok=True)
                filename = file_path + '/' + self.model_type + '/spk' + str(index+1) + '/' + files_name[index] + '.flac'
                write_wav(filename, s.unsqueeze(0).cpu(), 16000)
                print('saved in:', filename)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--mixed_sample', type=str, default='./samples/5ed8a9a769a27_5f885557c7f91.flac', help='Path to mix scp file.')
    parser.add_argument('-c', '--config', type=str, default='./configs/train_rnn.yml', help='Path to yaml file.')
    parser.add_argument('-w', '--weight', type=str, default='./weights/DualPath_RNN_179_-3.1895.pt', help="Path to model file.")
    parser.add_argument('-s', '--save_path', type=str, default='./samples', help='save result path')
    args = parser.parse_args()
    separator = Separation(args.config, args.weight, device='cuda')
    separator.separate(args.mixed_sample, args.save_path)
