import os

import torchaudio


def read_wav(fname, return_rate=False):
    '''
         Read wavfile using Pytorch audio
         input:
               fname: wav file path
               return_rate: Whether to return the sampling rate
         output:
                src: output tensor of size C x L 
                     L is the number of audio frames 
                     C is the number of channels. 
                sr: sample rate
    '''
    src, sr = torchaudio.load(fname, channels_first=True)
    if return_rate:
        return src.squeeze(), sr
    else:
        return src.squeeze()
    

def write_wav(fname, src, sample_rate):
    torchaudio.save(fname, src, sample_rate)


def get_file_name(file_path: str):
    return os.path.splitext(os.path.basename(file_path))[0]
