import math
import random

import torch as th
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from data.Dataset import Datasets
from utils.measure_time import measure_time 


class DiarizationDataset(pl.LightningDataModule):
    def __init__(self, data_root = './', total_percent = 1.0, train_percent = 0.75, valid_percent = 0.15, test_percent = 0.0, 
                 shuffle=False, num_workers=0, batch_size=1, pin_memory = False, 
                 sample_rate=8000, chunk_size=32000, least_size=16000, seed = 42):
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.least_size = least_size
        self.seed = seed
        self._set_seed(seed)
        self.g = th.Generator()
        self.g.manual_seed(seed)
        full_data_df = pd.read_csv(data_root) 
        full_data_df = full_data_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        assert math.isclose(train_percent + valid_percent + test_percent, 1.0, rel_tol=1e-9), "Sum doesnt equal to 1" 
        total_size = int(total_percent * len(full_data_df))
        full_data_df = full_data_df.iloc[:total_size]
        train_size = int(train_percent * len(full_data_df)) 
        val_size = int(valid_percent * len(full_data_df)) 
        test_size = len(full_data_df) - train_size - val_size
        self.train_df = full_data_df.iloc[:train_size] 
        self.val_df = full_data_df.iloc[train_size:train_size + val_size] 
        self.test_df = full_data_df.iloc[train_size + val_size:]
        
    @measure_time
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = Datasets(self.train_df,
                                        sample_rate = self.sample_rate,
                                        chunk_size = self.chunk_size,
                                        least_size = self.least_size)
            self.val_dataset = Datasets(self.val_df, 
                                        sample_rate = self.sample_rate,
                                        chunk_size = self.chunk_size,
                                        least_size = self.least_size)
        if stage in (None, "test"):
            self.test_dataset = Datasets(self.test_df,
                                         sample_rate = self.sample_rate,
                                         chunk_size = self.chunk_size,
                                         least_size = self.least_size)
            print(f"Size of test set: {len(self.test_dataset)}")
                    
    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_dataset,
                                    batch_size = self.batch_size,
                                    pin_memory = self.pin_memory,
                                    shuffle = self.shuffle,
                                    num_workers = self.num_workers,
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g)
        
    def val_dataloader(self):
        return th.utils.data.DataLoader(self.val_dataset,
                                    batch_size = self.batch_size,
                                    pin_memory = self.pin_memory,
                                    shuffle = False,
                                    num_workers = self.num_workers,
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g)
    
    def test_dataloader(self):
        return th.utils.data.DataLoader(self.test_dataset,
                                    batch_size = self.batch_size,
                                    pin_memory = self.pin_memory,
                                    shuffle = False,
                                    num_workers = self.num_workers,
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
#     import argparse
#     import sys
#     from utils.load_config import load_config  
#     print('start')
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-p", "--hparams", type=str, default="./configs/train_rnn.yml", help="hparams config file")
#     args, unknown = parser.parse_known_args()  # Игнорирует нераспознанные аргументы
#     cfg = load_config(args.hparams)

#     datamodule = DiarizationDataset(**cfg['data']).setup(stage = 'train')
#     dataloaders = {'train': datamodule.train_dataloader(), 'valid': datamodule.val_dataloader()}

#     # Получение первого батча данных из DataLoader
#     dataloader = dataloaders['train'] 
#     sample_mix, sample_refs = next(iter(dataloader))  
#     print(sample_mix)
#     print('chunks_num', len(sample_mix))
#     print(sample_mix[0])
#     print(sample_mix[0].shape)
#     print('----------------------------------------------')
#     print('spekears num', len(sample_refs))
#     print('firs_speaker list:', sample_refs[0])
#     print('chunks_nums', len(sample_refs[0]))
#     print(sample_refs[0][0].shape)