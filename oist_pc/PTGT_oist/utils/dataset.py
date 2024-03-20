#coding: utf-8
#----- Standard Library -----#
import re
import os
import sys
import random

#----- Public Package -----#
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
from PIL import Image
import cv2

#----- Module -----#
#None

PAD_ID = -1


##### OIST dataset #####
class OISTLoader(data.Dataset):
    def __init__(self, mode="train", data="GFP_immobile", length=16, transform=None):
        root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
        root_dir = os.path.join(*root_dir, "data")

        #--------------- 変更場所 ---------------#
        # 追加したフォルダの名前を入れる。辞書のKey部分には任意の名前を入れる。
        use_data_list = {
                        "GFP_immobile":
                            ["Mid_GFP_immobile1",
                            "Mid_GFP_immobile2",
                            "Mid_GFP_immobile3",
                            "Low_GFP_immobile1",
                            "Low_GFP_immobile2",
                            "Mid_GFP_v9"],
                        "SplitSample": # 任意の名前
                            ["aaa", # dataフォルダ内にある名前
                            "bbb",
                            "ccc",
                            "ddd",
                            "eee",
                            "fff"]}

        # 使用するframeの長さ
        use_frame_list = {
            "GFP_immobile": [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]],
            "SplitSample": [[0, 1], [0, 20], [0, 50], [0, 100], [0, 200], [0, 300]]}
        
        split_list = {"train": [0, 1, 2], "val": [3], "test": [4]}
        #次のように定義してもよい
        #split_list = {"train": [3, 4, 0, 1], "val": [2], "test": [4]}
        #--------------- 変更場所 ---------------#


        self.length = length

        use_paths = ["/" + os.path.join(root_dir, use_data_list[data][idx]) for idx in split_list[mode]]
        use_frames = [use_frame_list[data][idx] for idx in split_list[mode]]

        img_paths = []
        EP_paths = []
        for path, frame in zip(use_paths, use_frames):
            img_paths.append(sorted(glob.glob(path + "/video/*.png"))[frame[0] : frame[1]])
            EP_paths.append(path.split("/")[-1])
        
        # normalized, size => [1,H,W]
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        # get EP map
        EP_images = [h5py.File(f"/{root_dir}/Molecule_EP.hdf5", mode='r')[path] for path in EP_paths]
        EP_images = [list(EP.values())[frame[0] : frame[1]] for EP, frame in zip(EP_images, use_frames)]
        
        # The last image is not chosen.
        delete_index = [len(EP) for EP in EP_images]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]

        self.EP_images = [image[...] for EP in EP_images for image in EP]
        
        CP_data = [h5py.File(f"/{root_dir}/Cell_Point_Annotation.hdf5", mode='r')[path][...] for path in EP_paths]

        self.CP_data_per_data = [torch.tensor(data) for data in CP_data]

        self.CP_data = [sample_data[sample_data[:, 0] == frame] for sample_data in self.CP_data_per_data for frame in torch.unique(sample_data[:, 0])]
        self.CP_batch_data = torch.nn.utils.rnn.pad_sequence(self.CP_data, batch_first=True, padding_value=PAD_ID)
        self.CP_data = torch.cat(self.CP_data, dim=0)

        self.idx_transfer = list(range(len(self.EP_images)))
        
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        # 1iter => 0,1,2,3, 2iter => 4,5,6,7,...
        if mode == "val" or mode == "test":
            self.idx_transfer = [idx for idx in self.idx_transfer if idx%(length) == 0]

        # transform
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]image = np.concatenate([self.images[data_index + length] for length in range(self.length)], axis=0)
        image = np.concatenate([self.images[data_index + length] for length in range(self.length)], axis=0)
        label = np.concatenate([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        if self.transform:
            image, label, point = self.transform(image, label, point)

        return image[None], label, point

    def __len__(self):
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_data_per_data

##### OIST dataset #####
class OISTLoader_add(data.Dataset):
    def __init__(self, mode="train", data="GFP_immobile", length=16, transform=None):
        # split can be used only when 0
        root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
        root_dir = os.path.join(*root_dir, "data")

        #--------------- 変更場所 ---------------#
        # 追加したフォルダの名前を入れる。辞書のKey部分には任意の名前を入れる。
        use_data_list = {
                        "GFP_immobile":
                            ["Mid_GFP_immobile1",
                            "Mid_GFP_immobile2",
                            "Mid_GFP_immobile3",
                            "Low_GFP_immobile1",
                            "Low_GFP_immobile2"],
                        "SplitSample": # 任意の名前
                            ["aaa", # dataフォルダ内にある名前
                            "bbb",
                            "ccc",
                            "ddd",
                            "eee",
                            "fff"]}

        # 使用するframeの長さ
        use_frame_list = {
            "GFP_immobile": [[0, 100], [0, 100], [0, 100], [0, 100], [0, 100]],
            "SplitSample": [[0, 1], [0, 20], [0, 50], [0, 100], [0, 200]]}
        
        split_list = {"train": [0, 1, 2], "val": [3], "test": [4]}
        #次のように定義してもよい
        #split_list = {"train": [3, 4, 0, 1], "val": [2], "test": [4]}
        #--------------- 変更場所 ---------------#

        self.length = length

        use_paths = ["/" + os.path.join(root_dir, use_data_list[data][idx]) for idx in split_list[mode]]
        use_frames = [use_frame_list[data][idx] for idx in split_list[mode]]

        img_paths = []
        EP_paths = []
        for path, frame in zip(use_paths, use_frames):
            img_paths.append(sorted(glob.glob(path + "/video/*.png"))[frame[0] : frame[1]])
            EP_paths.append(path.split("/")[-1])
        
        CP_data = [h5py.File(f"/{root_dir}/Cell_Point_Annotation.hdf5", mode='r')[path][...] for path in EP_paths]

        self.CP_data_per_data = [torch.tensor(data) for data in CP_data]
        self.CP_data = torch.tensor(CP_data)

        self.CP_data = [sample_data[sample_data[:, 0] == frame] for sample_data in self.CP_data_per_data for frame in torch.unique(sample_data[:, 0])]
        self.CP_batch_data = torch.nn.utils.rnn.pad_sequence(self.CP_data, batch_first=True, padding_value=PAD_ID)
        self.CP_data = torch.cat(self.CP_data, dim=0)

        # normalized, size => [1,H,W]
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]

        # get EP map
        EP_images = [h5py.File(f"/{root_dir}/Molecule_EP.hdf5", mode='r')[path] for path in EP_paths]
        EP_images = [list(EP.values())[frame[0] : frame[1]] for EP, frame in zip(EP_images, use_frames)]
        
        # The last image is not chosen.
        delete_index = [len(EP) for EP in EP_images]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]

        self.EP_images = [image[...] for EP in EP_images for image in EP]
        
        
        self.idx_transfer = list(range(len(self.EP_images)))
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)
        
        if mode == "val" or mode == "test":
            # number of data % (length - 1) = 1
            over = len(self.images) % (length - 1)
            if over == 0:
                append_num = 1
            elif over == 1:
                append_num = 0
            else:
                append_num = 16 - over
            
            for _ in range(append_num):
                self.images.append(self.images[-1])
                self.EP_images.append(self.EP_images[-1])
                last_frame_data = self.CP_batch_data[-1:]
                last_frame_data[:, 0] += 1
                self.CP_batch_data = torch.cat([self.CP_batch_data, last_frame_data], dim=0)
            
            self.idx_transfer = list(range(len(self.EP_images)))[: -(length - 1)]

            # 1iter => 0,1,2,3, 2iter => 4,5,6,7,...
            self.idx_transfer = [idx for idx in self.idx_transfer if idx % (length - 1) == 0]

        # transform
        self.transform = transform

    def __getitem__(self, index):
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]
        image = np.concatenate([self.images[data_index + length] for length in range(self.length)], axis=0)
        label = np.concatenate([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        if self.transform:
            image, label, point = self.transform(image, label, point)

        return image[None], label, point

    def __len__(self):
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_data


def OISTLoader_raw_data(data="CD47_NA149_TMR_03.avi", frame=None):
    root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
    root_dir = "/" + os.path.join(*root_dir, "analysis_data")

    
    cap = cv2.VideoCapture(os.path.join(root_dir, data))

    if not cap.isOpened():
        # 正常に読み込めたのかチェックする
        # 読み込めたらTrue、失敗ならFalse
        print("動画の読み込み失敗")
        sys.exit()

    images = []
    
    while True:
        # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
        is_image, frame_img = cap.read()
        if is_image:
            # 画像を保存
            images.append(torch.from_numpy(frame_img.transpose(2,0,1)[[2,1,0]]))
        else:
            # フレーム画像が読込なかったら終了
            break
        if frame is not None:
            frame -= 1
            if frame < 0:
                break

        if len(images) == 16:
            img = torch.stack(images, dim=1)[:1] / 255
            images = [images[-1]]
            yield img[None]
    
    append_num = 16 - len(images)
    images = images + [images[-1] for _ in range(append_num)]

    img = torch.stack(images, dim=1)[:1] / 255
    yield img[None]


def OISTLoader_plot(data="CD47_NA149_TMR_03.avi", frame=None):
    root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
    root_dir = "/" + os.path.join(*root_dir, "analysis_data")

    video = torchvision.io.VideoReader(os.path.join(root_dir, data), "video")
    if frame is None:
        frame = 10 ** 10
    images = [video_data["data"] for video_data, _ in zip(video, range(frame))]
    img = torch.stack(images, dim=1)
    
    return img



def collate_delete_PAD(batch):
    image, label, point = list(zip(*batch))

    image = torch.stack(image)
    label = torch.stack(label)
    point = torch.stack(point)

    # Make PAD as low as possible.
    sorted_idx = torch.argsort(point[:, :, :, 0], dim=2, descending=True)
    point = torch.take_along_dim(point, sorted_idx[:, :, :, None], dim=2)
    max_point_num = (point[:, :, :, 0] != -1).sum(dim=-1).max()
    point = point[:, :, :max_point_num]

    return image, label, point

