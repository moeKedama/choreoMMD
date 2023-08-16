import os
import numpy as np
import json
import wave
import librosa
import fbx
import torchaudio
import torch
from torch.utils.data import Dataset


def getfolder(folder: str):
    root_folder = folder

    # 获取子文件夹列表
    subfolders = [subfolder for subfolder in os.listdir(root_folder) if
                  os.path.isdir(os.path.join(root_folder, subfolder))]

    # 打印子文件夹列表
    # print("子文件夹列表:")
    # print(subfolders)
    subfolders_dict = dict()
    # 获取各子文件夹中的文件列表
    for subfolder in subfolders:
        subfolders_dict[subfolder] = os.path.join(root_folder, subfolder)

    label_json_files = np.array(
        [os.path.join(subfolders_dict["label_json"], file) for file in os.listdir(subfolders_dict["label_json"]) if
         file.endswith('.json')])
    motion_fbx_files = np.array(
        [os.path.join(subfolders_dict["motion_fbx"], file) for file in os.listdir(subfolders_dict["motion_fbx"]) if
         file.endswith('.fbx')])
    music_clip_files = np.array(
        [os.path.join(subfolders_dict["music_clip"], file) for file in os.listdir(subfolders_dict["music_clip"]) if
         file.endswith('.wav')])

    return np.vstack((label_json_files, motion_fbx_files, music_clip_files)).transpose()


if __name__ == '__main__':
    choreo_folder = getfolder("ChoreoMaster_Dataset")
    target_duration = 2.0
    output_sample_rate = 16000
    target_num_samples = int(target_duration * output_sample_rate)
    total_files = choreo_folder.shape[0]
    # resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=output_sample_rate)
    for i in range(total_files):
        print(f"{i + 1} / {total_files}")
        # for x in choreo_folder:
        #     filename = x[2]
        _ = choreo_folder[i]
        filename = _[2]
        waveform, sample_rate = torchaudio.load(filename)
        # resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=output_sample_rate)

        # mode1 全部重采样 人耳听着区别不大
        if waveform.size(1) != 0:
            test_rate = int(sample_rate ** 2 / (waveform.size(1) // 2))  # 有一点抽象的推导
            resampled_waveform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=test_rate)(waveform)  # wip
        else:
            resampled_waveform = waveform

        if resampled_waveform.size(1) > target_num_samples:
            resampled_waveform = resampled_waveform[:, :target_num_samples]
        elif resampled_waveform.size(1) < target_num_samples:
            padding = target_num_samples - resampled_waveform.size(1)
            resampled_waveform = torch.nn.functional.pad(resampled_waveform, (0, padding))
        print(resampled_waveform.size(1))

        # mode2 低于2s进行padding,高于2s进行重采样
        # if waveform.size(1) <= target_num_samples:
        #     print("padding")
        #     padding = target_num_samples - waveform.size(1)
        #     resampled_waveform = torch.nn.functional.pad(waveform, (0, padding))
        # else:
        #     print("resample")
        #     test_rate = int(sample_rate ** 2 / (waveform.size(1) // 2))  # 有一点抽象的推导
        #     resampled_waveform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=test_rate)(waveform)  # wip
        #
        #     if resampled_waveform.size(1) > target_num_samples:
        #         print("cut")
        #         resampled_waveform = resampled_waveform[:, :target_num_samples]
        #     elif resampled_waveform.size(1) < target_num_samples:
        #         print("resample padding 1")
        #         padding = target_num_samples - resampled_waveform.size(1)
        #         resampled_waveform = torch.nn.functional.pad(resampled_waveform, (0, padding))

        save_path = os.path.join("Processed_Dataset\\music_clip", filename.split("\\")[-1])
        torchaudio.save(filepath=save_path, src=resampled_waveform, sample_rate=test_rate)
