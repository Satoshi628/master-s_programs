#coding: utf-8
#----- Standard Library -----#
import os
import glob

#----- Public Package -----#
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms.functional as F
import cv2

#----- Module -----#
#None


##### Konmo dataset #####
class Konmo_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/konmo", mode="train", transform=None):
        # 30FPS => 120FPS case delay=4

        image_paths = sorted(glob.glob(os.path.join(root_dir, "data2", "images", "*")))
        label_paths = sorted(glob.glob(os.path.join(root_dir, "data2", "labels", "*")))
        
        #画像サイズが一致しているかチェック
        img_path_temp = []
        lab_path_temp = []
        for img_path, lab_path in zip(image_paths, label_paths):
            img_size = cv2.imread(img_path).shape[-2:]
            lab_size = cv2.imread(lab_path).shape[-2:]
            if (img_size[0] == lab_size[0]) and (img_size[1] == lab_size[1]):
                img_path_temp.append(img_path)
                lab_path_temp.append(lab_path)
        image_paths = img_path_temp
        label_paths = lab_path_temp
        
        if mode == "train":
            image_paths = image_paths[: int(0.7 * len(image_paths))]
            label_paths = label_paths[: int(0.7 * len(label_paths))]
        else:
            image_paths = image_paths[int(0.7 * len(image_paths)):]
            label_paths = label_paths[int(0.7 * len(label_paths)):]
        self.images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for path in image_paths]
        self.labels = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) // 255 for path in label_paths]
        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数

        image = self.images[index]
        label = self.labels[index]

        image = torch.from_numpy(image).float() / 255.
        label = torch.from_numpy(label) 

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)


##### Konmo dataset #####
class Konmo_Video_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/konmo",split=512, crop=[1500,0,1250,1000], data_num=None, transform=None):
        self.split = split
        self.crop = crop

        if data_num is None:
            image_paths = sorted(glob.glob(os.path.join(root_dir, "P1270204", "*")))
        else:
            image_paths = sorted(glob.glob(os.path.join(root_dir, "P1270204", "*")))[:data_num]


        self.images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[:, crop[1]:crop[3], crop[0]:crop[2]] for path in image_paths]

        #transform
        self.transform = transform


    # 画像&ラベル読み込み
    def __getitem__(self, index):
        #shuffle=Trueならindexはランダムな数

        image = self.images[index]

        image = torch.from_numpy(image).float() / 255.

        if self.transform:
            image, _ = self.transform(image, image)
        
        #画像を縦で分割する
        #元画像の大きさは(分割数+1)*split//2
        #splitの大きさで割り切れるようにpadding
        split_size = self.split//2
        H = (image.shape[1] % (split_size))
        if H != 0:
            H = (split_size) - H
        image = F.pad(image, padding=[0, 0, 0, H])

        images = []
        for idx in range((image.shape[1] // (split_size) ) - 1):
            images.append(image[:,split_size*idx:split_size*(idx+1)+256])
        images = torch.stack(images)
        return images


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.images)



##### Zenigoke dataset #####
class Zenigoke_Loader(data.Dataset):
    # 初期設定
    def __init__(self, root_dir="/mnt/kamiya/dataset/root/data", mode="train", split=0, length=16, delay=2, transform=None):
        # 30FPS => 120FPS case delay=4
        
        split_list = [{"train": [1, 2, 3, 4], "val": [5], "test": [6]},
                      {"train": [6, 1, 2, 3], "val": [4], "test": [5]},
                      {"train": [5, 6, 1, 2], "val": [3], "test": [4]},
                      {"train": [4, 5, 6, 1], "val": [2], "test": [3]},
                      {"train": [3, 4, 5, 6], "val": [1], "test": [2]}]
        
        self.length = length
        self.delay = delay
        
        #パス、使用フレームを選択
        use_paths = [os.path.join(root_dir, f"sample{idx}") for idx in split_list[split][mode]]

        img_paths = []
        EP_paths = []
        CP_path = []
        path_length = []
        for path in use_paths:
            #相対パスを追加
            img_paths.append(sorted(glob.glob(os.path.join(path, "video", "*.png"))))
            path_length.append(len(glob.glob(os.path.join(path, "video", "*.png"))))
            #imgパスからCPパスを取得する
        
        self.images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for paths in img_paths for path in paths]
        
        if mode == "test":
            self.video_file = glob.glob(os.path.join(use_paths[0], "*.avi"))[0]
        
        #EP保存
        EP_images = [np.load(os.path.join(path, "EPmaps.npz"))['arr_0'] for path in use_paths]
        EP_images = np.concatenate(EP_images, axis=0)

        self.EP_images = EP_images

        #Cell Point保存
        CP_data = [torch.from_numpy(np.loadtxt(os.path.join(path, "det.txt"), delimiter=',')).float()
                    for path in use_paths]
        
        
        self.CP_batch_data = [Points[Points[:,0] == idx] for video_len, Points in zip(path_length, CP_data) for idx in range(video_len)]
        #track idの被りを無くす
        #track id の最初は1
        track_id_max = 0
        for idx in range(len(CP_data)):
            CP_data[idx][:, 1] += track_id_max
            track_id_max = CP_data[idx][:, 1].max()


        #削除するindexを求めるlength-1コ消す
        delete_index = [frame for frame in path_length]
        delete_index = [sum(delete_index[: idx + 1]) - delete for idx in range(len(delete_index)) for delete in range(length - 1, 0, -1)]
        
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
        image = np.stack([self.images[data_index + length] for length in range(self.length) if length %self.delay == 0], axis=1)
        label = np.stack([self.EP_images[data_index + length] for length in range(self.length)], axis=0)
        point = [self.CP_batch_data[data_index + length].clone() for length in range(self.length)]

        image = torch.from_numpy(image).float() / 255.
        label = torch.from_numpy(label) 

        if self.transform:
            image, label, point = self.transform(image, label, point)

        label = label.transpose(0, 1)
        return image, label, point


    # 1epochで処理する枚数の設定(データの長さ分読み込んだらloaderが終了する)
    def __len__(self):
        # データの長さ
        return len(self.idx_transfer)

    def get_track(self):
        return self.CP_batch_data

    def get_videos(self):
        return self.images

    def get_EP_images(self):
        return self.EP_images


def collate_delete_PAD(batch):
    image, label, point = list(zip(*batch))

    image = torch.stack(image)
    label = torch.stack(label)

    return image, label, point
