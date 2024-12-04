import numpy as np
import torchaudio
import torch

from utils.funtctional import handle_scp
from utils.data_processing import read_wav
from torch.utils.data import DataLoader as Loader

def make_dataloader(opt):
    # make train's dataloader
    
    train_dataset = Datasets(
        opt['datasets']['train']['dataroot_mix'],
        [opt['datasets']['train']['dataroot_targets'][0],
         opt['datasets']['train']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    train_dataloader = Loader(train_dataset,
                              batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                              num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                              shuffle=opt['datasets']['dataloader_setting']['shuffle'])
    
    # make validation dataloader
    
    val_dataset = Datasets(
        opt['datasets']['val']['dataroot_mix'],
        [opt['datasets']['val']['dataroot_targets'][0],
         opt['datasets']['val']['dataroot_targets'][1]],
        **opt['datasets']['audio_setting'])
    val_dataloader = Loader(val_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            shuffle=False)
    
    return train_dataloader, val_dataloader


class AudioReader(object):
    def __init__(self, scp_path, sample_rate=8000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())

    def _load(self, key):
        src, sr = read_wav(self.index_dict[key], return_rate=True)
        if self.sample_rate is not None and sr != self.sample_rate:
            raise RuntimeError('SampleRate mismatch: {:d} vs {:d}'.format(
                sr, self.sample_rate))
        return src

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        for key in self.keys:
            yield key, self._load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError('Unsupported index type: {}'.format(type(index)))
        if type(index) == int:
            num_uttrs = len(self.keys)
            if num_uttrs < index and index < 0:
                raise KeyError('Interger index out of range, {:d} vs {:d}'.format(
                    index, num_uttrs))
            index = self.keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))

        return self._load(index)
    

class Datasets(torch.utils.data.Dataset):
    def __init__(self, mix_scp=None, ref_scp=None, sample_rate=8000, chunk_size=32000, least_size=16000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(
            mix_scp, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio
        self.ref_audio = [AudioReader(
            r, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio for r in ref_scp]

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        return self.mix_audio[index], [ref[index] for ref in self.ref_audio]