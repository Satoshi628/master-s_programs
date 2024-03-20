#coding: utf-8
#----- Standard Library -----#
import re
import os
import sys
import random
import glob
import math

#----- Public Package -----#
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
import cv2

#----- Module -----#
#None

PAD_ID = -100

##### OIST dataset #####
class OIST_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov_HighFPS", Staining="GFP", mode="train", split=0, length=16, delay=4, transform=None):
        # 30FPS => 120FPS case delay=4
        use_data_list = {
                        "GFP-Mid":
                            ["GFP-Mid-1/video",
                            "GFP-Mid-2/video",
                            "GFP-Mid-3/video",
                            "GFP-Mid-4/video",
                            "GFP-Mid-5/video"],
                        "GFP_Low-Mid":
                            ["GFP_Low-Mid-1/video",
                            "GFP_Low-Mid-2/video",
                            "GFP_Low-Mid-3/video",
                            "GFP_Low-Mid-4/video",
                            "GFP_Low-Mid-5/video"]}
        
        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 0, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 0], "val": [1], "test": [2]},
                      {"train": [2, 3, 4], "val": [0], "test": [1]},
                      {"train": [1, 2, 3], "val": [4], "test": [0]}]
        
        self.length = length
        self.delay = delay
        
        #エラー対策
        if not Staining in use_data_list:  #キーがあるかどうか
            raise KeyError("Staningが{0},{1}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, use_data_list[Staining][idx]) for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        CP_path = []
        path_length = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(path + "/*.png")))
            path_length.append(len(glob.glob(path + "/*.png")))
            #imgパスからCPパスを取得する
            EP_paths.append(path.split("/")[-2])
            CP_path.append(path.split("/")[-2])
        
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #EP保存
        EP_images = [h5py.File("{}/Molecule_EP.hdf5".format(root_dir), mode='r')[path] for path in EP_paths]
        EP_images = [list(EP.values()) for EP in EP_images]
        self.EP_images = [image[...] for EP in EP_images for image in EP]

        #Cell Point保存
        CP_data = [h5py.File("{}/Cell_Point_Annotation.hdf5".format(root_dir), mode='r')[path][...] for path in CP_path]

        #削除するindexを求めるlength-1コ消す
        delete_index = [frame for frame in path_length]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])

        #track idの被りを無くす
        #track id の最初は1
        track_id_max = 0
        for idx in range(len(CP_data)):
            CP_data[idx][:, 1] += track_id_max
            track_id_max = CP_data[idx][:, 1].max()

        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data)
        
        #frameごとにわけ、batchに対応できるようにする
        #cell_max_count * frameの配列に入れるための番号を作製
        #もし一つもdetectionのないframeが存在したらバグる。まあないから大丈夫
        frame_idx = np.ones(self.CP_data.shape[0])
        frame_idx[0] = 0
        frame_idx[np.cumsum(count[:-1], axis=0)] += np.max(count) - count[:-1]
        frame_idx = np.cumsum(frame_idx, axis=0, dtype=np.int32)
        
        self.CP_batch_data = PAD_ID * torch.ones([len(self.images) * np.max(count), 4])
        self.CP_batch_data[frame_idx] = self.CP_data
        self.CP_batch_data = self.CP_batch_data.view(len(self.images), np.max(count), 4)

        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(len(self.images)))
        #最後のidxから削除
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        #val,testは同じ画像を入力する必要がない追跡アルゴリズムの場合
        #1iter 0,1,2,3, 2iter 4,5,6,7,...とする
        if mode == "val" or mode == "test":
            self.idx_transfer = [idx for idx in self.idx_transfer if idx%(length) == 0]
        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]
        image = np.concatenate([self.images[data_index + length] for length in range(self.length) if length %self.delay == 0], axis=0)
        label = np.concatenate([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        if self.transform:
            image, label, point = self.transform(image, label, point)

        return image[None], label[None], point


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_batch_data

    def get_max_move_speed(self):
        track_ids = np.unique(self.CP_batch_data[:, :, 1])
        track_ids = track_ids[track_ids != PAD_ID]
        max_value_diff1 = 0
        max_value_diff2 = 0
        for track_id in track_ids:
            track = self.CP_batch_data[self.CP_batch_data[:, :, 1] == track_id][:, 2:]
            if track.shape[0] > 1:
                dist = np.diff(track, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff1:
                    max_value_diff1 = max_dist

            if track.shape[0] > 2:
                dist = np.diff(track, n=2, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff2:
                    max_value_diff2 = max_dist
        print(max_value_diff1.item())
        print(max_value_diff2.item())
        #R = math.sqrt(2 * 0.3 / 120) * 2 * 1000 / 50
        #heoretical_value = math.sqrt(R ** 2 + 60 * math.sqrt(2) / 120 * R + 2 * (30 / 120) ** 2)
        #print(theoretical_value)
        #return theoretical_value
        
        #print(max_value)
        return [max_value_diff1.item(), max_value_diff2.item()]

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_images


##### OIST dataset #####
class OIST_Loader_Train30(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/Sim_immobile_Density_mov", Staining="GFP", mode="train", split=0, length=16, delay=2, transform=None):
        # 30FPS => 120FPS case delay=4
        use_data_list = {
                        "GFP-Mid":
                            ["Sample1",
                            "Sample2",
                            "Sample3",
                            "Sample4",
                            "Sample5"],
                        "TMR-Mid":
                            ["Mid_01",
                            "Mid_02",
                            "Mid_03",
                            "Mid_04",
                            "Mid_05"],
                        "SF650-Mid":
                            ["Mid_01",
                             "Mid_02",
                             "Mid_03",
                             "Mid_04",
                             "Mid_05"]}
        
        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 0, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 0], "val": [1], "test": [2]},
                      {"train": [2, 3, 4], "val": [0], "test": [1]},
                      {"train": [1, 2, 3], "val": [4], "test": [0]}]
        
        self.length = length
        self.delay = delay
        
        #エラー対策
        if not Staining in use_data_list:  #キーがあるかどうか
            raise KeyError("Staningが{0},{1}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, use_data_list[Staining][idx]) for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        CP_path = []
        path_length = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(os.path.join(path, "video", "*.png"))))
            path_length.append(len(glob.glob(os.path.join(path, "video", "*.png"))))
            #imgパスからCPパスを取得する
            EP_paths.append(path.split("/")[-1])
            CP_path.append(path.split("/")[-1])
        
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #EP保存
        EP_images = [np.load(os.path.join(root_dir, path, "EPmaps.npz"))['arr_0'] for path in EP_paths]
        EP_images = np.concatenate(EP_images, axis=0)

        self.EP_images = EP_images

        #Cell Point保存
        CP_data = [np.loadtxt(os.path.join(root_dir, path, "det.txt"), delimiter=',') for path in CP_path]

        #削除するindexを求めるlength-1コ消す
        delete_index = [frame for frame in path_length]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])

        #track idの被りを無くす
        #track id の最初は1
        track_id_max = 0
        for idx in range(len(CP_data)):
            CP_data[idx][:, 1] += track_id_max
            track_id_max = CP_data[idx][:, 1].max()

        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data, dtype=torch.float)
        
        #frameごとにわけ、batchに対応できるようにする
        #cell_max_count * frameの配列に入れるための番号を作製
        #もし一つもdetectionのないframeが存在したらバグる。まあないから大丈夫
        frame_idx = np.ones(self.CP_data.shape[0])
        frame_idx[0] = 0
        frame_idx[np.cumsum(count[:-1], axis=0)] += np.max(count) - count[:-1]
        frame_idx = np.cumsum(frame_idx, axis=0, dtype=np.int32)
        
        self.CP_batch_data = PAD_ID * torch.ones([len(self.images) * np.max(count), 4])
        self.CP_batch_data[frame_idx] = self.CP_data
        self.CP_batch_data = self.CP_batch_data.view(len(self.images), np.max(count), 4)

        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(len(self.images)))
        #最後のidxから削除
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        #val,testは同じ画像を入力する必要がない追跡アルゴリズムの場合
        #1iter 0,1,2,3, 2iter 4,5,6,7,...とする
        if mode == "val" or mode == "test":
            self.idx_transfer = [idx for idx in self.idx_transfer if idx%(length) == 0]
        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]
        image = np.concatenate([self.images[data_index + length] for length in range(self.length) if length %self.delay == 0], axis=0)
        label = np.stack([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        if self.transform:
            image, label, point = self.transform(image, label, point)

        label = label.transpose(0, 1)
        return image[None], label, point


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_batch_data

    def get_max_move_speed(self):
        track_ids = np.unique(self.CP_batch_data[:, :, 1])
        track_ids = track_ids[track_ids != PAD_ID]
        max_value_diff1 = 0
        max_value_diff2 = 0
        for track_id in track_ids:
            track = self.CP_batch_data[self.CP_batch_data[:, :, 1] == track_id][:, 2:]
            if track.shape[0] > 1:
                dist = np.diff(track, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff1:
                    max_value_diff1 = max_dist

            if track.shape[0] > 2:
                dist = np.diff(track, n=2, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff2:
                    max_value_diff2 = max_dist
        print(max_value_diff1.item())
        print(max_value_diff2.item())
        #R = math.sqrt(2 * 0.3 / 120) * 2 * 1000 / 50
        #heoretical_value = math.sqrt(R ** 2 + 60 * math.sqrt(2) / 120 * R + 2 * (30 / 120) ** 2)
        #print(theoretical_value)
        #return theoretical_value
        
        #print(max_value)
        return [max_value_diff1.item(), max_value_diff2.item()]

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_images



##### OIST dataset #####
class OIST_Loader_Train30_2(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/Sim_immobile_Density_mov", Staining="GFP", mode="train", split=0, length=16, delay=2, transform=None):
        # 30FPS => 120FPS case delay=4
        use_data_list = {
                        "GFP-Mid":
                            ["Sample1",
                            "Sample2",
                            "Sample3",
                            "Sample4",
                            "Sample5"],
                        "GFP_Low-Mid":
                            ["GFP_Low-Mid-1",
                            "GFP_Low-Mid-2",
                            "GFP_Low-Mid-3",
                            "GFP_Low-Mid-4",
                            "GFP_Low-Mid-5"]}
        
        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 0, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 0], "val": [1], "test": [2]},
                      {"train": [2, 3, 4], "val": [0], "test": [1]},
                      {"train": [1, 2, 3], "val": [4], "test": [0]}]
        
        self.length = length
        self.delay = delay
        
        #エラー対策
        if not Staining in use_data_list:  #キーがあるかどうか
            raise KeyError("Staningが{0},{1}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, use_data_list[Staining][idx]) for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        CP_path = []
        path_length = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(os.path.join(path, "video", "*.png"))))
            path_length.append(len(glob.glob(os.path.join(path, "video", "*.png"))))
            #imgパスからCPパスを取得する
            EP_paths.append(path.split("/")[-1])
            CP_path.append(path.split("/")[-1])
        
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #EP保存
        EP_images = [np.load(os.path.join(root_dir, path, "EPmaps.npz"))['arr_0'] for path in EP_paths]
        EP_images = np.concatenate(EP_images, axis=0)
        multi_EP_images = [np.load(os.path.join(root_dir, path, "multi_EPmaps.npz"))['arr_0'] for path in EP_paths]
        multi_EP_images = np.concatenate(multi_EP_images, axis=0)

        self.EP_images = np.concatenate([EP_images, multi_EP_images], axis=1)

        #Cell Point保存
        CP_data = [np.loadtxt(os.path.join(root_dir, path, "det.txt"), delimiter=',') for path in CP_path]

        #削除するindexを求めるlength-1コ消す
        delete_index = [frame for frame in path_length]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])

        #track idの被りを無くす
        #track id の最初は1
        track_id_max = 0
        for idx in range(len(CP_data)):
            CP_data[idx][:, 1] += track_id_max
            track_id_max = CP_data[idx][:, 1].max()

        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data, dtype=torch.float)
        
        #frameごとにわけ、batchに対応できるようにする
        #cell_max_count * frameの配列に入れるための番号を作製
        #もし一つもdetectionのないframeが存在したらバグる。まあないから大丈夫
        frame_idx = np.ones(self.CP_data.shape[0])
        frame_idx[0] = 0
        frame_idx[np.cumsum(count[:-1], axis=0)] += np.max(count) - count[:-1]
        frame_idx = np.cumsum(frame_idx, axis=0, dtype=np.int32)
        
        self.CP_batch_data = PAD_ID * torch.ones([len(self.images) * np.max(count), 4])
        self.CP_batch_data[frame_idx] = self.CP_data
        self.CP_batch_data = self.CP_batch_data.view(len(self.images), np.max(count), 4)

        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(len(self.images)))
        #最後のidxから削除
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        #val,testは同じ画像を入力する必要がない追跡アルゴリズムの場合
        #1iter 0,1,2,3, 2iter 4,5,6,7,...とする
        if mode == "val" or mode == "test":
            self.idx_transfer = [idx for idx in self.idx_transfer if idx%(length) == 0]
        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]
        image = np.concatenate([self.images[data_index + length] for length in range(self.length) if length %self.delay == 0], axis=0)
        label = np.stack([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        if self.transform:
            image, label, point = self.transform(image, label, point)

        label = label.transpose(0, 1)
        return image[None], label, point


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_batch_data

    def get_max_move_speed(self):
        track_ids = np.unique(self.CP_batch_data[:, :, 1])
        track_ids = track_ids[track_ids != PAD_ID]
        max_value_diff1 = 0
        max_value_diff2 = 0
        for track_id in track_ids:
            track = self.CP_batch_data[self.CP_batch_data[:, :, 1] == track_id][:, 2:]
            if track.shape[0] > 1:
                dist = np.diff(track, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff1:
                    max_value_diff1 = max_dist

            if track.shape[0] > 2:
                dist = np.diff(track, n=2, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff2:
                    max_value_diff2 = max_dist
        print(max_value_diff1.item())
        print(max_value_diff2.item())
        #R = math.sqrt(2 * 0.3 / 120) * 2 * 1000 / 50
        #heoretical_value = math.sqrt(R ** 2 + 60 * math.sqrt(2) / 120 * R + 2 * (30 / 120) ** 2)
        #print(theoretical_value)
        #return theoretical_value
        
        #print(max_value)
        return [max_value_diff1.item(), max_value_diff2.item()]

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_images



##### OIST dataset #####
class OIST_Loader_Test30FPS(data.Dataset):
    
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST_new/Sim_immobile_Density_mov", Staining="GFP", mode="train", split=0, length=16, delay=2, transform=None):
        # 30FPS => 120FPS case delay=4
        use_data_list = {
                        "GFP-Mid":
                            ["Sample1",
                            "Sample2",
                            "Sample3",
                            "Sample4",
                            "Sample5"],
                        "TMR-Mid":
                            ["Mid_01",
                            "Mid_02",
                            "Mid_03",
                            "Mid_04",
                            "Mid_05"],
                        "SF650-Mid":
                            ["Mid_01",
                             "Mid_02",
                             "Mid_03",
                             "Mid_04",
                             "Mid_05"]}
        
        split_list = [{"val": [3], "test": [4]},
                      {"val": [2], "test": [3]},
                      {"val": [1], "test": [2]},
                      {"val": [0], "test": [1]},
                      {"val": [4], "test": [0]}]
        
        self.length = length
        self.delay = delay
        
        #エラー対策
        if not Staining in use_data_list:  #キーがあるかどうか
            raise KeyError("Staningが{0},{1}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, use_data_list[Staining][idx]) for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        CP_path = []
        path_length = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(os.path.join(path, "video", "*.png"))))
            path_length.append(len(glob.glob(os.path.join(path, "video", "*.png"))))
            #imgパスからCPパスを取得する
            EP_paths.append(path.split("/")[-1])
            CP_path.append(path.split("/")[-1])
        
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]
        
        #EP保存
        EP_images = [np.load(os.path.join(root_dir, path, "EPmaps.npz"))['arr_0'] for path in EP_paths]
        EP_images = np.concatenate(EP_images, axis=0)

        self.EP_images = EP_images

        #Cell Point保存
        CP_data = [np.loadtxt(os.path.join(root_dir, path, "det.txt"), delimiter=',') for path in CP_path]

        #削除するindexを求めるlength-1コ消す
        delete_index = [frame for frame in path_length]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
        #frameのかぶりがある可能性があるため結合前にcountする
        count = np.concatenate([np.unique(data[:, 0], return_counts=True)[-1] for data in CP_data])

        #track idの被りを無くす
        #track id の最初は1
        track_id_max = 0
        for idx in range(len(CP_data)):
            CP_data[idx][:, 1] += track_id_max
            track_id_max = CP_data[idx][:, 1].max()

        #結合
        self.CP_data = np.concatenate(CP_data)
        self.CP_data = torch.tensor(self.CP_data, dtype=torch.float)
        
        #frameごとにわけ、batchに対応できるようにする
        #cell_max_count * frameの配列に入れるための番号を作製
        #もし一つもdetectionのないframeが存在したらバグる。まあないから大丈夫
        frame_idx = np.ones(self.CP_data.shape[0])
        frame_idx[0] = 0
        frame_idx[np.cumsum(count[:-1], axis=0)] += np.max(count) - count[:-1]
        frame_idx = np.cumsum(frame_idx, axis=0, dtype=np.int32)
        
        self.CP_batch_data = PAD_ID * torch.ones([len(self.images) * np.max(count), 4])
        self.CP_batch_data[frame_idx] = self.CP_data
        self.CP_batch_data = self.CP_batch_data.view(len(self.images), np.max(count), 4)

        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(0, len(self.images), length))
        #transform
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        # label.size() => [length,H,W]
        # point.size() => [length,mol_max_number,4(frame,id,x,y)]
        images = []
        labels = []
        points = []
        for leng in range(self.length):
            idx = min(data_index + leng, len(self.images) - 1)
            images.append(self.images[idx])
            labels.append(self.EP_images[idx])
            points.append(self.CP_batch_data[idx])

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        points = torch.stack(points, dim=0)


        if self.transform:
            images, labels, points = self.transform(images, labels, points)

        return images[None], labels[None], points


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_batch_data

    def get_max_move_speed(self):
        track_ids = np.unique(self.CP_batch_data[:, :, 1])
        track_ids = track_ids[track_ids != PAD_ID]
        max_value_diff1 = 0
        max_value_diff2 = 0
        for track_id in track_ids:
            track = self.CP_batch_data[self.CP_batch_data[:, :, 1] == track_id][:, 2:]
            if track.shape[0] > 1:
                dist = np.diff(track, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff1:
                    max_value_diff1 = max_dist

            if track.shape[0] > 2:
                dist = np.diff(track, n=2, axis=0)
                dist = dist ** 2
                dist = np.sqrt(dist.sum(axis=-1))
                max_dist = dist.max()
                if max_dist > max_value_diff2:
                    max_value_diff2 = max_dist
        print(max_value_diff1.item())
        print(max_value_diff2.item())
        #R = math.sqrt(2 * 0.3 / 120) * 2 * 1000 / 50
        #heoretical_value = math.sqrt(R ** 2 + 60 * math.sqrt(2) / 120 * R + 2 * (30 / 120) ** 2)
        #print(theoretical_value)
        #return theoretical_value
        
        #print(max_value)
        return [max_value_diff1.item(), max_value_diff2.item()]


    def get_videos(self):
        return self.images




##### OIST dataset #####
class OIST_Loader_Exp(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/OIST/Experimental_data", Staining="CD47-GFP_01.avi", length=16):
        # 30FPS => 120FPS case delay=4
        self.length = length

        cap = cv2.VideoCapture(os.path.join(root_dir, Staining))
        
        if not cap.isOpened():
            # 正常に読み込めたのかチェックする
            # 読み込めたらTrue、失敗ならFalse
            print("動画の読み込み失敗")
            sys.exit()
        
        self.images = []
        while True:
            # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
            is_image, frame_img = cap.read()
            if is_image:
                # BGR => Gray
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)[None]
                self.images.append(torch.from_numpy(frame_img).float()/255)
            else:
                # フレーム画像が読込なかったら終了
                break
        
        path_length = len(self.images)
        
        #削除するindexを求めるlength-1コ消す
        delete_index = list(range(path_length))
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
        self.idx_transfer = list(range(0, len(self.images), length))


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        # image.size() => [length,H,W]
        images = []
        for length in range(self.length):
            idx = min(data_index + length, len(self.images) - 1)
            images.append(self.images[idx])
        images = torch.stack(images, dim=1)
        
        return images


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_videos(self):
        return self.images



##### PTC dataset #####
class PTC_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/PTC", Molecule="RECEPTOR", Density="Low", mode="train", split=0, length=16,add_mode=False, transform=None):
        use_data_list = {
                        "Low":
                            ["Sample1",
                            "Sample2",
                            "Sample3",
                            "Sample4",
                            "Sample5"],
                        "Mid":
                            ["Sample6",
                            "Sample7",
                            "Sample8",
                            "Sample9",
                            "Sample10"],
                        "High":
                            ["Sample11",
                            "Sample12",
                            "Sample13",
                            "Sample14",
                            "Sample15"],
                        "Mid2":
                            ["Sample16",
                            "Sample17",
                            "Sample18",
                            "Sample19",
                            "Sample20"]}

        split_list = [{"train": [0, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 0, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 0], "val": [1], "test": [2]},
                      {"train": [2, 3, 4], "val": [0], "test": [1]},
                      {"train": [1, 2, 3], "val": [4], "test": [0]}]
        
        #エラー対策
        if not Density in use_data_list:  # キーがあるかどうか
            raise KeyError("Densityが{0},{1},{2}以外の文字になっています。".format(*list(use_data_list.keys())))
        if not mode in split_list[0]:  # キーがあるかどうか
            raise KeyError("modeが{0},{1},{2}以外の文字になっています。".format(*list(split_list[0].keys())))

        self.length = length

        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, Molecule, use_data_list[Density][idx]) for idx in split_list[split][mode]]
        use_frames = 0

        
        img_paths = []
        EP_paths = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(path + "/*.tif")))
            #imgパスからmpmパスを取得する
            EP_paths.append(re.search(r'Sample\d+', path).group())
        
        #読み込み時すべてnumpyより、list内部numpyの方が処理が早い
        #image保存、正規化も行う(0~1の値にする)、大きさ[1,H,W]となる。
        self.images = [np.array(Image.open(path).convert("L"))[None] / 255 for paths in img_paths for path in paths]

        #EP保存
        EP_images = [h5py.File("{}/{}/Molecule_EP.hdf5".format(root_dir, Molecule), mode='r')[path] for path in EP_paths]
        EP_images = [list(EP.values()) for EP in EP_images]
        
        #削除するindexを求めるlength-1コ消す
        delete_index = [len(EP) for EP in EP_images]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]

        #EP_imageをすべてメモリに保存する。
        self.EP_images = [image[...] for EP in EP_images for image in EP]
        
        #Cell Point保存
        CP_data = [h5py.File("{}/{}/Cell_Point_Annotation.hdf5".format(root_dir, Molecule), mode='r')[path][...] for path in EP_paths]

        self.CP_data_per_data = [torch.tensor(data) for data in CP_data]
        #結合
        self.CP_data = [sample_data[sample_data[:, 0] == frame] for sample_data in self.CP_data_per_data for frame in torch.unique(sample_data[:, 0])]
        self.CP_batch_data = torch.nn.utils.rnn.pad_sequence(self.CP_data, batch_first=True, padding_value=PAD_ID)
        self.CP_data = torch.cat(self.CP_data,dim=0)


        #getitemのindexとEP_igamesのindexを合わせる配列の作製
        self.idx_transfer = list(range(len(self.EP_images)))
        #最後のidxから削除
        for idx in delete_index[::-1]:
            self.idx_transfer.pop(idx)

        #val,testは同じ画像を入力する必要がない追跡アルゴリズムの場合
        #1iter 0,1,2,3, 2iter 4,5,6,7,...とする
        if mode == "val" or mode == "test":
            length_overlap = add_mode * 1
            self.idx_transfer = [idx for idx in self.idx_transfer if idx % (length - length_overlap) == 0]

        #transforme
        self.transform = transform

    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数
        data_index = self.idx_transfer[index]
        
        #imageは[length,H,W]、labelは[length,H,W]、pointは[length,mol_max_number,4(frame,id,x,y)]
        image = np.concatenate([self.images[data_index + length] for length in range(self.length)], axis=0)
        label = np.concatenate([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = torch.stack([self.CP_batch_data[data_index + length] for length in range(self.length)], dim=0)

        # もしself.transformが有効なら
        if self.transform:
            image, label, point = self.transform(image, label, point)

        return image[None], label, point  #image[1,length,H,W],label[length,H,W],point[length,num,4(frame,id,x,y)]

    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_data_per_data



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

if __name__ == "__main__":
    import transforms as tf
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    train_dataset = OIST_Loader_Train30_2(
        Staining="GFP-Mid", mode="train", split=0, length=16, delay=4, transform=train_transform)
    max_move = train_dataset.get_max_move_speed()
    input(max_move)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=8,
        shuffle=True,
        #num_workers=2,
        drop_last=True)
    for batch_idx, (inputs, targets, point) in enumerate(train_loader):
        print(inputs.shape)
        print(targets.shape)
        print(point.shape)
        input()
