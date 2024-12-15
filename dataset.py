import os
import random
import math
from glob import glob
from typing import List
from tqdm import tqdm 

import torch as th
import torchaudio
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import numpy as np

from utils.measure_time import measure_time 


class AudioDataset(th.utils.data.Dataset):
    
    def __init__(self, mix_file_paths: List[str], ref_file_paths: List[List[str]], 
                 sr: int = 8000, chunk_size: int = 32000, least_size: int = 16000):
        super().__init__()
        self.mix_audio = []
        self.ref_audio = []
        k = len(ref_file_paths[0])
        for mix_path, ref_paths in zip(mix_file_paths, ref_file_paths):
            chunked_mix = self._load_audio(mix_path, sr, chunk_size, least_size)
            if not chunked_mix: continue
            ref_audio_chunks = [self._load_audio(ref_path, sr, chunk_size, least_size) for ref_path in ref_paths]
            if k != len(ref_audio_chunks): continue
            self.mix_audio.append(chunked_mix)
            self.ref_audio.append(ref_audio_chunks)

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, idx):
        mix = self.mix_audio[idx]
        refs = self.mix_audio[idx]
        # refs = [ref[idx] for ref in self.ref_audio]
        return mix, refs
    
    @staticmethod
    def _load_audio(path:str, sr: int, chunk_size: int, least_size: int):
        audio, _sr = torchaudio.load(path)
        if _sr != sr: raise RuntimeError(f"Sample rate mismatch: {_sr} vs {sr}")
        if audio.shape[-1] < least_size: return []
        audio_chunks = []
        if least_size < audio.shape[-1] < chunk_size:
            pad_size = chunk_size - audio.shape[-1]
            audio_chunks.append(F.pad(audio, (0, pad_size), mode='constant'))
        else:
            start = 0
            while start + chunk_size <= audio.shape[-1]:
                audio = audio.squeeze() # warning
                audio_chunks.append(audio[start:start + chunk_size])
                start += least_size
        return audio_chunks 


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, csv_file:bool = False, total_percent: float = 0.1, train_percent:float = 0.8, valid_percent:float = 0.1, 
                 test_percent: float = 0.1, num_workers: int = 4, batch_size: int = 512, pin_memory = False, seed: int = 42, 
                 sample_rate: int = 8000, chunk_size: int = 32000, least_size: int = 16000):
        super().__init__()
        self.batch_size = batch_size
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.least_size = least_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self._set_seed(seed)
        self.g = th.Generator()
        self.g.manual_seed(seed)
        self.mix_paths = []
        self.ref_paths = []
        
        if csv_file:
            full_df = pd.read_csv(data_dir)
            for _, row in full_df.iterrows():
                self.mix_paths.append (row.iloc[0])
                self.ref_paths.append(sorted([row[column] for column in full_df.columns[1:]]))
        else:
            mixed_list = glob(os.path.join(data_dir, "*.flac"))
            for mx in tqdm(mixed_list):
                mx = mx.replace('\\', '/')
                self.mix_paths.append (mx)
                mx_df = pd.read_csv(mx.replace('flac', 'csv'))
                f_real = sorted([mx_df.iloc[0][column] for column in mx_df.columns[1:]])
                self.ref_paths.append(f_real)

        self.mix_paths = self.mix_paths[:int(len(self.mix_paths) * total_percent)]
        self.ref_paths = self.ref_paths[:int(len(self.ref_paths) * total_percent)]
        random.shuffle(self.mix_paths)
        assert math.isclose(train_percent + valid_percent + test_percent, 1.0, rel_tol=1e-9), "Sum doesnt equal to 1" 
        self.train_len = int(len(self.mix_paths) * train_percent)
        self.valid_len = int(len(self.mix_paths) * valid_percent)
        self.test_len = int(len(self.mix_paths) * test_percent)

    @measure_time
    def setup(self, stage = 'train'):
        assert stage in ['train', 'eval'], "Invalid stage"
        
        if stage == 'train': 
            self.train_dataset = AudioDataset(self.mix_paths[:self.train_len], 
                                              self.ref_paths[:self.train_len], 
                                              sr = self.sr, 
                                              chunk_size = self.chunk_size, 
                                              least_size = self.least_size)
            print(f"Size of training set: {len(self.train_dataset)}")
            
            self.val_dataset = AudioDataset(self.mix_paths[self.train_len:self.train_len + self.valid_len], 
                                            self.ref_paths[self.train_len:self.train_len + self.valid_len], 
                                            sr = self.sr, 
                                            chunk_size = self.chunk_size, 
                                            least_size = self.least_size) 
            print(f"Size of validation set: {len(self.val_dataset)}")

        if stage == 'eval':
            self.test_dataset = AudioDataset(self.mix_paths[self.train_len + self.valid_len:], 
                                             self.ref_paths[self.train_len + self.valid_len:], 
                                             sr = self.sr, 
                                             chunk_size = self.chunk_size, 
                                             least_size = self.least_size)
            print(f"Size of test set: {len(self.test_dataset)}")

        return self
        
    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_dataset, 
                                        batch_size=self.batch_size, 
                                        pin_memory = self.pin_memory,
                                        shuffle=True, 
                                        num_workers=self.num_workers,
                                        worker_init_fn=self.seed_worker,
                                        generator=self.g)

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.val_dataset, 
                                        batch_size=self.batch_size, 
                                        pin_memory = self.pin_memory,
                                        shuffle=False, 
                                        num_workers=self.num_workers,
                                        worker_init_fn=self.seed_worker,
                                        generator=self.g)

    def test_dataloader(self):
        return th.utils.data.DataLoader(self.test_dataset, 
                                        batch_size=self.batch_size,
                                        pin_memory = self.pin_memory, 
                                        shuffle=False, 
                                        num_workers=self.num_workers, 
                                        worker_init_fn=self.seed_worker,
                                        generator=self.g)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    def seed_worker(self, worker_id):
        worker_seed = th.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


# if __name__ == '__main__':
#     from utils.load_config import load_config 
#     cfg = load_config('./configs/train_rnn.yml')
#     datamodule = AudioDataModule(**cfg['data']).setup(stage = 'train')
#     dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}