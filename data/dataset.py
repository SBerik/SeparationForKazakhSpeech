import torch

from data.AudioData import AudioReader


class Datasets(torch.utils.data.Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
       chunk_size (int, optional): split audio size (default: 32000(4 s))
       least_size (int, optional): Minimum split size (default: 16000(2 s))
    '''
    def __init__(self, df=None, sample_rate=16000, chunk_size=32000, least_size=16000):
        super(torch.utils.data.Dataset, self).__init__()
        k = len(df.iloc[0]) - 2
        mix_scp = []
        ref_scp = [[] for _ in range (k)]
        for _, row in df.iterrows():
            common_len_idx = row['common_len_idx']
            mix_scp.append([common_len_idx, row['mixed_audio']])
            i = 0
            for col in df.columns[2:]:
                audio_value = row[col]
                ref_scp[i].append([common_len_idx, audio_value])
                i += 1 
        self.mix_audio = AudioReader(mix_scp, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio
        self.ref_audio = [AudioReader(r, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio for r in ref_scp]
    
    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        return self.mix_audio[index], [ref[index] for ref in self.ref_audio]


# class ExtendedAudioDataset(Datasets):
#     def __getitems__(self, item):
#         return self.__getitem__(item) 

# if __name__ == "__main__":
#     dataset = Datasets("/",
#                       ["", ""])
#     for i in dataset.mix_audio:
#         if i.shape[0] != 32000:
#             print('fail')
