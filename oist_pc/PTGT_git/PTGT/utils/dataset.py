#coding: utf-8
#----- Standard Library -----#
import re
import os
import random
import glob

#----- Public Package -----#
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image

#----- Module -----#
#None

PAD_ID = -1


##### OIST dataset #####
class OISTLoader(data.Dataset):
    def __init__(self, mode="train", split=0, length=16, transform=None):
        # split can be used only when 0
        root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
        root_dir = os.path.join(*root_dir, "data")

        use_data_list = {
                        "GFP":
                            ["Density_Mid_GFP1/video",
                            "Density_Mid_GFP2/video",
                            "Density_Low_GFP1/video"]}

        use_frame_list = {
            "GFP": [[0, 100], [0, 100], [0, 100]]}
        
        split_list = [{"train": [0,1], "val": [1], "test": [2]}]

        self.length = length

        use_paths = ["/" + os.path.join(root_dir, use_data_list["GFP"][idx]) for idx in split_list[split][mode]]
        
        use_frames = [use_frame_list["GFP"][idx] for idx in split_list[split][mode]]
        
        img_paths = []
        EP_paths = []
        for path, frame in zip(use_paths, use_frames):
            img_paths.append(sorted(glob.glob(path + "/*.png"))[frame[0] : frame[1]])
            EP_paths.append(path.split("/")[-2])
        
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
    def __init__(self, mode="train", split=0, length=16, transform=None):
        # split can be used only when 0
        root_dir = [path for path in os.path.dirname(__file__).split("/")[:-1]]
        root_dir = os.path.join(*root_dir, "data")

        use_data_list = {
                        "GFP":
                            ["Density_Mid_GFP1/video",
                            "Density_Mid_GFP2/video",
                            "Density_Low_GFP1/video"]}

        use_frame_list = {
            "GFP": [[0, 100], [0, 100], [0, 100]]}
        
        split_list = [{"train": [0,1], "val": [1], "test": [2]}]


        self.length = length

        use_paths = ["/" + os.path.join(root_dir, use_data_list["GFP"][idx]) for idx in split_list[split][mode]]
        use_frames = [use_frame_list["GFP"][idx] for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        for path, frame in zip(use_paths, use_frames):
            img_paths.append(sorted(glob.glob(path + "/*.png"))[frame[0] : frame[1]])
            EP_paths.append(path.split("/")[-2])
        
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

