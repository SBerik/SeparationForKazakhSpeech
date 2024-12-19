from data.Dataset import Datasets
from utils.measure_time import measure_time 
import torch as th
import random
import numpy as np
import math
import pandas as pd

class DiarizationDataset:
    def __init__(self, data_root = './', train_percent = 0.75, valid_percent = 0.15, test_percent = 0.0, 
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
        assert math.isclose(train_percent + valid_percent + test_percent, 1.0, rel_tol=1e-9), "Sum doesnt equal to 1" 
        train_size = int(train_percent * len(full_data_df)) 
        val_size = int(valid_percent * len(full_data_df)) 
        test_size = len(full_data_df) - train_size - val_size
        self.train_df = full_data_df.iloc[:train_size] 
        self.val_df = full_data_df.iloc[train_size:train_size + val_size] 
        self.test_df = full_data_df.iloc[train_size + val_size:]
        
    @measure_time
    def setup(self, stage = 'train'):
        assert stage in ['train', 'eval'], "Invalid stage" 
        if stage == 'train': 
            self.train_dataset = Datasets(self.train_df,
                                        sample_rate = self.sample_rate,
                                        chunk_size = self.chunk_size,
                                        least_size = self.least_size)
            print(f"Size of training set: {len(self.train_dataset)}")
            
            self.val_dataset = Datasets(self.val_df, 
                                        sample_rate = self.sample_rate,
                                        chunk_size = self.chunk_size,
                                        least_size = self.least_size)
            print(f"Size of validation set: {len(self.val_dataset)}")
        # To Do 
        # self.test_dataset
        
        return self # warning! 
        
    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_dataset,
                                    batch_size = self.batch_size,
                                    pin_memory = self.pin_memory,
                                    shuffle = self.shuffle,
                                    num_workers = self.num_workers,
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g)
        
    def val_dataloader(self):
        return th.utils.data.DataLoader(self.train_dataset,
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
        
    # ToDo
    # def test_dataloader(self):


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