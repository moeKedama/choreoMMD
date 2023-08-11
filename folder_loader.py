import os
import numpy as np
import json
import wave
import librosa
import fbx
# from fbx import *
# from samples import FbxCommon

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import npz_loader


# 指定根文件夹路径
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


def json_loader(path):
    with open(path, "rb") as json_file:
        data = json.load(json_file)
    return data


# def wav_loader(path):
#     with wave.open(path, 'rb') as wav_file:
#         # 获取音频文件的基本信息
#         # num_channels = wav_file.getnchannels()  # 通道数
#         # sample_width = wav_file.getsampwidth()  # 采样宽度（字节数）
#         # frame_rate = wav_file.getframerate()  # 采样率
#         num_frames = wav_file.getnframes()  # 帧数
#
#         # 读取音频数据
#         audio_data = wav_file.readframes(num_frames)
#
#     return audio_data

def librosa_loader(path):
    audio_data, sample_rate = librosa.load(path, sr=16000)
    return audio_data, sample_rate


def audio2logmelspectrogram(audio_data, sample_rate, n_mels=96, hop_length=160):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


class MMDDataset(Dataset):
    def __init__(self, dataset, root_dir, labels=None, indices=None):
        self.datas = getfolder(root_dir)
        self.dataset = dataset

        imgs = [item for item in self.datas]
        self.labels = np.array(labels)
        # self.labels = np.array(labels) / 100
        self.x = imgs
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        self.jsonLoader = json_loader
        # self.wavLoader = wav_loader
        self.wavLoader = librosa_loader
        # self.fbxLoader = fbx_loader

    def __getitem__(self, index):
        index = self.indices[index]
        json_raw = self.jsonLoader(self.x[index][0])
        name = json_raw['name']
        music_style1 = json_raw['music_style1']
        music_style2 = json_raw['music_style2']
        motion_style1 = json_raw['motion_style1']
        motion_style2 = json_raw['motion_style2']
        signature = json_raw['signature']

        wav_raw, sample_rate = self.wavLoader(self.x[index][2])

        spectrogram = audio2logmelspectrogram(wav_raw, sample_rate, n_mels=96, hop_length=160)

        # img = self.input_trans(self.loader(self.x[index]))

        return index, name, music_style1, music_style2, motion_style1, motion_style2, signature, wav_raw, spectrogram

    def __len__(self):
        return len(self.indices)


# 2s在16k采样率下一共32000个点，8s在16k下一共128000个点
# 数据为1~3s，方法1是插值，方法2是截断和0填充

if __name__ == '__main__':
    # choreo_folder = getfolder("ChoreoMaster_Dataset")
    # imgs = [item for item in choreo_folder]

    choreo_dataset = MMDDataset(dataset="MMD", root_dir="ChoreoMaster_Dataset")

    a = choreo_dataset.__getitem__(0)
    b = choreo_dataset.__len__()

    # manager, scene = FbxCommon.InitializeSdkObjects()
    # filename = "0071@000.fbx"
    # result = FbxCommon.LoadScene(manager, scene, filename)
    # dir(scene)
    # # root_node = scene.GetRootNode()
    # pNode = scene.GetRootNode()
    # dir(pNode)
    # camera = pNode.GetCamera()
    # dir(camera)
    #
    # for i in range(scene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))):
    #     # Take 001 遍历 take
    #     lAnimStack = scene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
    #     print("Take: %s" % lAnimStack.GetName())
    #
    #     lAnimLayer = lAnimStack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), i)
    #
    #     lAnimCurve = pNode.LclRotation.GetCurve(lAnimLayer, "X")
    #     if (lAnimCurve is not None):
    #         lKeyCount = lAnimCurve.KeyGetCount()
    #         for lCount in range(lKeyCount):
    #             lKeyValue = lAnimCurve.KeyGetValue(lCount)
    #             lKeyTime = lAnimCurve.KeyGetTime(lCount).GetSecondDouble()
    #
    #     lAnimCurve = pNode.LclRotation.GetCurve(lAnimLayer, "Y")
    #     if (lAnimCurve is not None):
    #         lKeyCount = lAnimCurve.KeyGetCount()
    #         for lCount in range(lKeyCount):
    #             lKeyValue = lAnimCurve.KeyGetValue(lCount)
    #             lKeyTime = lAnimCurve.KeyGetTime(lCount).GetSecondDouble()
    #
    #     lAnimCurve = pNode.LclRotation.GetCurve(lAnimLayer, "Z")
    #     if (lAnimCurve is not None):
    #         lKeyCount = lAnimCurve.KeyGetCount()
    #         for lCount in range(lKeyCount):
    #             lKeyValue = lAnimCurve.KeyGetValue(lCount)
    #             lKeyTime = lAnimCurve.KeyGetTime(lCount).GetSecondDouble()

    # ii = []
    # for i in range(2708):
    #     # print(choreo_dataset.__getitem__(i)[7].shape)
    #     print(choreo_dataset.__getitem__(i)[-1])
    #     if choreo_dataset.__getitem__(i)[-1] == (0,):
    #         ii.append(i)
    # audio_data, sample_rate = librosa.load("0071@001.wav", sr=16000)
    # #
    # n_mels = 96
    # hop_length = 160
    # mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    #
    # # 将梅尔频谱转换为对数幅度
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
