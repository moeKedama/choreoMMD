import librosa
import numpy as np
from datautil.folder_loader import getfolder
from datautil.folder_loader import audio2logmelspectrogram
import torchaudio.transforms as transforms
from matplotlib import pyplot as plt


# 好像从频谱出和从音频出结果不太一样, 还是从音频出好了, 从频谱出还存在神秘的问题
def onset_features(wav, sample_rate=16000, n_mels=96, hop_length=160):
    onset = librosa.onset.onset_strength(y=wav, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    onset_slice = onset[:200]
    return onset_slice


def onset_features_mel(spectrogram, n_mels=96, hop_length=160):
    return librosa.onset.onset_strength(S=spectrogram, n_mels=n_mels, hop_length=hop_length)


def energy_features(wav, hop_length=160):
    rms = librosa.feature.rms(y=wav, hop_length=hop_length)
    rms_slice = rms[:, :200]
    return rms_slice


def energy_features_mel(spectrogram, hop_length=160, frame_length=191):
    return librosa.feature.rms(S=spectrogram, frame_length=frame_length, hop_length=hop_length)


if __name__ == '__main__':
    print("wip")
    choreo_folder = getfolder("../Processed_Dataset")

    total_files = choreo_folder.shape[0]
    # resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=output_sample_rate)
    for i in range(total_files):
        # print(f"{i + 1} / {total_files}")
        # for x in choreo_folder:
        #     filename = x[2]
        _ = choreo_folder[i]
        filename = _[2]
        waveform, sample_rate = librosa.load(filename, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=96,
                                                         hop_length=160)
        mel_spectrogram_cut = mel_spectrogram[:, :200]

        rms_mel = energy_features_mel(mel_spectrogram_cut, frame_length=190)
        rms = energy_features(waveform)
        # onset_mel = onset_features_mel(mel_spectrogram_cut)

        # logmelspectrogram = audio2logmelspectrogram(waveform, sample_rate)
        #
        # onset_feature = librosa.onset.onset_strength(y=waveform, sr=sample_rate, n_mels=96, hop_length=160)
        # onset_feature_cut = onset_feature[:200]
        #
        # for _ in onset_feature:
        #     if _.item() != 0.0:
        #         print("onset")

        # rms_curve = librosa.feature.rms(y=waveform, hop_length=160)

        # 绘制 RMS 能量曲线
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(rms_mel, sr=sample_rate, x_axis='time')
        plt.title("RMS Energy Curve1")
        plt.xlabel("Time")
        plt.ylabel("RMS Energy")
        plt.show()

        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(rms, sr=sample_rate, x_axis='time')
        plt.title("RMS Energy Curve2")
        plt.xlabel("Time")
        plt.ylabel("RMS Energy")
        plt.show()
