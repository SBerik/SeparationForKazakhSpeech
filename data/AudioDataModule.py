import os
import random
import math
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch as th
import pytorch_lightning as pl

from utils.measure_time import measure_time
from AudioDataset import AudioDataset

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
                self.ref_paths.append([row.iloc[1], sorted([row[column] for column in full_df.columns[2:]])])
        else: 
            mixed_list = glob(os.path.join(data_dir, "*.flac"))
            for mx in tqdm(mixed_list):
                mx = mx.replace('\\', '/')
                self.mix_paths.append (mx)
                mx_df = pd.read_csv(mx.replace('flac', 'csv'))
                f_real = [mx_df.iloc[0], sorted([mx_df.iloc[0][column] for column in mx_df.columns[2:]])] # THERE IS BAG need to fixed 
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

        return self # warning
        
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


# to run this file - python3 -m datasets.data_loader.py

# if __name__ == '__main__':
#     from utils.load_config import load_config 
#     cfg = load_config('./configs/train_rnn.yml')
#     datamodule = AudioDataModule(**cfg['data']).setup(stage = 'train')
#     dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}