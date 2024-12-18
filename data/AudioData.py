import torch.nn.functional as F
import torch
import torchaudio
import sys
sys.path.append('../')

def handle_scp(scp_path):
    '''
    Read scp file script
    input: 
          scp_path: .scp file's file path
    output: 
          scp_dict: {'key':'wave file path'}
    '''
    scp_dict = dict()
    line = 0
    lines = open(scp_path, 'r').readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError("For {}, format error in line[{:d}]: {}".format(
                scp_path, line, scp_parts))
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                key, scp_path))

        scp_dict[key] = value

    return scp_dict

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
    '''
         Write wav file
         input:
               fname: wav file path
               src: frames of audio
               sample_rate: An integer which is the sample rate of the audio
         output:
               None
    '''
    torchaudio.save(fname, src, sample_rate)


class AudioReader(object):
    '''
        Class that reads Wav format files
        Input:
            scp_path (str): a different scp file address
            sample_rate (int, optional): sample rate (default: 8000)
            chunk_size (int, optional): split audio size (default: 32000(4 s))
            least_size (int, optional): Minimum split size (default: 16000(2 s))
        Output:
            split audio (list)
    '''

    def __init__(self, scp_path, sample_rate=8000, chunk_size=32000, least_size=16000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())
        print(self.keys[0])
        self.audio = []
        self.chunk_size = chunk_size
        self.least_size = least_size
        self.split()

    def split(self):
        '''
            split audio with chunk_size and least_size
        '''
        for key in self.keys:
            utt = read_wav(self.index_dict[key])
            if utt.shape[0] < self.least_size:
                continue
            if utt.shape[0] > self.least_size and utt.shape[0] < self.chunk_size:
                gap = self.chunk_size-utt.shape[0]
                self.audio.append(F.pad(utt, (0, gap), mode='constant'))
            if utt.shape[0] >= self.chunk_size:
                start = 0
                while True:
                    if start + self.chunk_size > utt.shape[0]:
                        break
                    self.audio.append(utt[start:start+self.chunk_size])
                    start += self.least_size

    def get_num_after_splitting(self):
        print(len(self.audio))


if __name__ == "__main__":
    a = AudioReader('F:/ISSAI_KSC2_unpacked/diahard_data/scp_files_k=2/tr_mix.scp', 
                    sample_rate=16000, 
                    chunk_size=32000, 
                    least_size=16000)
    
    ref_scp = ['F:/ISSAI_KSC2_unpacked/diahard_data/scp_files_k=2/tr_s1.scp', 
               'F:/ISSAI_KSC2_unpacked/diahard_data/scp_files_k=2/tr_s2.scp']
    
    b = [AudioReader(r, sample_rate=16000, chunk_size=32000, least_size=16000).audio for r in ref_scp]
    
    audio = a.audio
    print(len(audio))
