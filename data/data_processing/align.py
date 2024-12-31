import os
import numpy as np
import soundfile as sf
from scipy.signal import resample


# Abosolute normalization, ex-name: normalize_audio
#  RMS стал равным 1
def abs_normalize_audio(audio):
    """Нормализация аудиосигнала по энергии RMS."""
    rms = np.sqrt(np.mean(audio**2))
    return audio / rms if rms > 0 else audio


def relative_normalize_audio(signals):
    energies = [np.sum(s ** 2) for s in signals]
    max_energy = max(energies)
    scaling_factors = [np.sqrt(max_energy / energy) for energy in energies]
    normalized_audios = [s * scale for s, scale in zip(signals, scaling_factors)]
    mixed_audio = np.sum(normalized_audios, axis=0)
    return mixed_audio, normalized_audios

# Для уселение и уменьшение одного из сигналов. 
# snr_range=(0, 5)
# snr = np.random.uniform(*snr_range)
def mix_signals(signal1, signal2, snr):
    """Смешивание двух сигналов с заданным SNR."""
    # Нормализация сигналов
    signal1 = normalize_audio(signal1)
    signal2 = normalize_audio(signal2)

    # Вычисление коэффициента масштабирования для второго сигнала
    scale = 10 ** (-snr / 20)  # Преобразование SNR в линейный масштаб
    signal2_scaled = signal2 * scale

    # Суммирование сигналов
    mixed_signal = signal1 + signal2_scaled
    return mixed_signal, signal1, signal2_scaled


def create_mixed_dataset(input_dir, output_dir, snr_range=(0, 5), sample_rate=8000):
    """
    Args:
        input_dir: Папка с исходными WAV-файлами.
        output_dir: Папка для сохранения смешанных данных.
        snr_range: Диапазон SNR (в дБ).
        sample_rate: Частота дискретизации для всех аудиофайлов.
    """
    os.makedirs(output_dir, exist_ok=True)
    wav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.wav')]

    # Проверка наличия файлов
    if len(wav_files) < 2:
        raise ValueError("Недостаточно файлов для смешивания. Нужны минимум два.")

    for i, file1 in enumerate(wav_files):
        for file2 in wav_files[i+1:]:
            signal1, sr1 = sf.read(file1)
            signal2, sr2 = sf.read(file2)

            if sr1 != sample_rate:
                signal1 = resample(signal1, int(len(signal1) * sample_rate / sr1))
            if sr2 != sample_rate:
                signal2 = resample(signal2, int(len(signal2) * sample_rate / sr2))

            min_length = min(len(signal1), len(signal2))
            signal1 = signal1[:min_length]
            signal2 = signal2[:min_length]

            snr = np.random.uniform(*snr_range)

            mixed_signal, clean1, clean2 = mix_signals(signal1, signal2, snr)

            base_name1 = os.path.basename(file1).replace('.flac', '')
            base_name2 = os.path.basename(file2).replace('.flac', '')
            output_base = f"{base_name1}_{base_name2}_SNR{snr:.1f}dB"

            sf.write(os.path.join(output_dir, f"{output_base}_mix.wav"), mixed_signal, sample_rate)
            sf.write(os.path.join(output_dir, f"{output_base}_clean1.wav"), clean1, sample_rate)
            sf.write(os.path.join(output_dir, f"{output_base}_clean2.wav"), clean2, sample_rate)
            print(f"Смешанный файл сохранён: {output_base}_mix.wav")
