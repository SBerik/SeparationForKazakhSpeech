import os
import argparse
import tqdm

import torch 
from models.model_rnn import Dual_RNN_model
from utils.load_config import load_config 
import torchaudio

def get_file_name(file_path: str):
    return os.path.splitext(os.path.basename(file_path))[0]

def read_wav(fname, return_rate=False):
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()

def write_wav(fname, src, sample_rate):
    torchaudio.save(fname, src, sample_rate)


class Separation():
    def __init__(self, mixed_sample, yaml_path, weight, gpuid = 0):
        super(Separation, self).__init__()
        self.name = get_file_name (mixed_sample)
        self.mix = read_wav(mixed_sample)
        cfg = load_config(yaml_path)
        self.net = Dual_RNN_model(**cfg['model'])
        self.net.to('cpu')
        dicts = torch.load(weight, map_location='cpu')
        self.net.load_state_dict(dicts['model_state_dict'])
        self.gpuid = gpuid
    
    def inference(self, file_path):
        self.net.eval()
        with torch.no_grad():
            egs=self.mix
            norm = torch.norm(egs,float('inf'))
            if len(self.gpuid) != 0:
                if egs.dim() == 1:
                    egs = torch.unsqueeze(egs, 0)
                ests=self.net(egs)
                spks=[torch.squeeze(s.detach().cpu()) for s in ests]
            else:
                if egs.dim() == 1:
                    egs = torch.unsqueeze(egs, 0)
                ests=self.net(egs)
                print(ests[0].shape)
                spks=[torch.squeeze(s.detach()) for s in ests]
            index=0
            for s in spks:
                # Normalize
                s = s - torch.mean(s)
                s = s * norm / torch.max(torch.abs(s))
                index += 1
                os.makedirs(file_path + '/spk' + str(index), exist_ok=True)
                filename = file_path + '/spk' + str(index) + '/' + self.name + '_inferenced.flac'
                write_wav(filename, s.unsqueeze(0), 16000)
                print('saved in:', filename)


def main(args):
    gpuid = [int(i) for i in args.gpuid.split(',')]
    separation = Separation(args.mixed_sample, args.config, args.weight, gpuid)
    separation.inference(args.save_path)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-m', '--mixed_sample', type=str, default='./samples/f1_inform_1051_00_f1_inform_1082_00.flac', help='Path to mix scp file.')
    parser.add_argument('-c', '--config', type=str, default='./configs/train_rnn.yml', help='Path to yaml file.')
    parser.add_argument('-w', '--weight', type=str, default='./weights/Dual_Path_RNN_49_-1.2763.pt', help="Path to model file.")
    parser.add_argument('-g', '--gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument('-s', '--save_path', type=str, default='./samples', help='save result path')
    main(parser.parse_args())