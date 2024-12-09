import librosa
import numpy as np 
from tqdm.notebook import tqdm 
import pandas as pd
from predictor import get_VAD


# Idea 1.
# Почему мы выбираем этот вариант. Т. к. для модель может начать думать что тишина это тоже разговор speech    
def overlap_using_min(audios, sr=16000):
    loaded_audios = [librosa.load(audio, sr=sr)[0] for audio in audios]
    min_length = min(len(audio) for audio in loaded_audios)
    trimmed_audios = [audio[:min_length] for audio in loaded_audios]
    overlay = np.sum(trimmed_audios, axis=0)
    return overlay


def overlap(audio_1: str, audio_2: str, sr=16000):    
    # # Idea 2.
    # silence_threshold = 1e-5  
    # is_silent_1 = np.all(np.abs(y1[-int(sr*0.1):]) < silence_threshold)
    # is_silent_2 = np.all(np.abs(y2[-int(sr*0.1):]) < silence_threshold)
    # if len(y1) < len(y2):
    #     y1 = np.pad(y1, (0, len(y2) - len(y1))) if is_silent_1 else y1
    # else:
    #     y2 = np.pad(y2, (0, len(y1) - len(y2))) if is_silent_2 else y2

    # # Idea 3.
    # if len(y1) < len(y2):
    #     y1 = np.pad(y1, (0, len(y2) - len(y1)))
    # else:
    #     y2 = np.pad(y2, (0, len(y1) - len(y2)))

    overlay = y1 + y2
    return overlay


def get_trimmed_sample (t, sr, base_path):
    audio, sr = librosa.load(t, sr=sr)
    predict_csv = get_VAD(t, base_path, sr=sr, model = False)
    df = pd.read_csv(predict_csv)
    start = df[df['speech'] == 1].iloc[0]['start_time']
    end = df[df['speech'] == 1].iloc[-1]['end_time']
    trimmed_audio = audio[int(start * sr):int(end * sr)]
    return trimmed_audio