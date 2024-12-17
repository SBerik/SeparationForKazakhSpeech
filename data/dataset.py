from typing import List

import torch as th
import torchaudio
import torch.nn.functional as F


class AudioDataset(th.utils.data.Dataset):
    def __init__(self, mix_file_paths: List[str], ref_file_paths: List[List[str]], 
                 sr: int = 8000, chunk_size: int = 32000, least_size: int = 16000):
        super().__init__()
        self.mix_audio = []
        self.ref_audio = []
        k = len(ref_file_paths[1])
        for mix_path, ref_paths in zip(mix_file_paths, ref_file_paths):
            common_len = ref_paths[0]
            chunked_mix = self._load_audio(mix_path, sr, common_len, chunk_size, least_size)
            if not chunked_mix: continue
            ref_audio_chunks = []    
            for ref_path in ref_paths[1]:
                res = self._load_audio(ref_path, sr, common_len, chunk_size, least_size)
                ref_audio_chunks.append(res)
            if k != len(ref_audio_chunks): continue
            self.mix_audio.append(chunked_mix)
            self.ref_audio.append(ref_audio_chunks)

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, idx):
        mix = self.mix_audio[idx][0]
        refs = self.ref_audio[idx][0]
        # refs = [ref[idx] for ref in self.ref_audio]
        return mix, refs
    
    @staticmethod
    def _load_audio(path:str, sr: int, common_len: int, chunk_size: int, least_size: int):
        audio, _sr = torchaudio.load(path)
        audio = audio[:, :common_len]
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


class ExtendedAudioDataset(AudioDataset):
    def __getitems__(self, item):
        return self.__getitem__(item) 