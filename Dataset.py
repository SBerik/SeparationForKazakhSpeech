import os
import random
import math
from glob import glob
import torch as th
import torchaudio
import pytorch_lightning as pl
from typing import Optional, List
import torch.nn.functional as F
import pandas as pd

EPS = 1e-8


class AudioDataset(th.utils.data.Dataset):
    def __init__(self, mix_paths: List[str], ref_paths: List[List[str]], 
                 sample_rate: int = 8000, chunk_size: int = 32000, least_size: int = 16000):
        super().__init__()
        self.mix_audio = self._load_audio(mix_paths, sample_rate, chunk_size, least_size)
        self.ref_audio = [self._load_audio(ref, sample_rate, chunk_size, least_size) for ref in ref_paths]

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, idx):
        mix = self.mix_audio[idx]
        refs = [ref[idx] for ref in self.ref_audio]
        return mix, refs

    @staticmethod
    def _load_audio(paths: List[str], sample_rate: int, chunk_size: int, least_size: int):
        audios = []
        for path in paths:
            audio, sr = torchaudio.load(path)
            if sr != sample_rate:
                raise RuntimeError(f"Sample rate mismatch: {sr} vs {sample_rate}")

            # Pad or chunk the audio
            if audio.shape[-1] < least_size:
                continue
            if least_size <= audio.shape[-1] < chunk_size:
                pad_size = chunk_size - audio.shape[-1]
                audios.append(F.pad(audio, (0, pad_size), mode='constant'))
            else:
                start = 0
                while start + chunk_size <= audio.shape[-1]:
                    audios.append(audio[:, start:start + chunk_size])
                    start += least_size
        return audios


class AudioDataModule(pl.LightningDataModule):
    def __init__(self, mix_dir: str, ref_dirs: List[str], batch_size: int = 128, 
                 train_split: float = 0.8, val_split: float = 0.1, sample_rate: int = 8000, 
                 chunk_size: int = 32000, least_size: int = 16000, num_workers: int = 4, seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.least_size = least_size
        self.num_workers = num_workers
        self.seed = seed

        self.mix_paths = glob(os.path.join(mix_dir, "*.flac"))

        # Сделать так что бы аудидорожки были одинаковой длины
        # Отсортировать что бы было как мы объеденяли.  
        ref_paths = []
        for mx_sample in self.mix_paths:
            mx_sample = mx_sample.replace('\\', '/')
            mx_csv = mx_sample.replace('flac', 'csv')
            mx_df = pd.read_csv(mx_csv)
            
            ref_paths.append(sorted(glob(os.path.join(ref_dir, "*.flac"))))

        self.ref_paths = [glob(os.path.join(ref_dir, "*.flac")) for ref_dir in ref_dirs]

        random.seed(seed)
        random.shuffle(self.mix_paths)

        total = len(self.mix_paths)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)

        self.train_paths = self.mix_paths[:train_end]
        self.val_paths = self.mix_paths[train_end:val_end]
        self.test_paths = self.mix_paths[val_end:]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = AudioDataset(self.train_paths, 
                                          self.ref_paths, 
                                          self.sample_rate, 
                                          self.chunk_size, 
                                          self.least_size)
        
        self.val_dataset = AudioDataset(self.val_paths, 
                                        self.ref_paths, 
                                        self.sample_rate, 
                                        self.chunk_size, 
                                        self.least_size)
        
        self.test_dataset = AudioDataset(self.test_paths, 
                                         self.ref_paths, 
                                         self.sample_rate, 
                                         self.chunk_size, 
                                         self.least_size)

    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=True, 
                                        num_workers=self.num_workers)

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.val_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        num_workers=self.num_workers)

    def test_dataloader(self):
        return th.utils.data.DataLoader(self.test_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        num_workers=self.num_workers)
