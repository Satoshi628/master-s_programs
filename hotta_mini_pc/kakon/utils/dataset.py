#coding: utf-8
#----- Standard Library -----#
import os
import sys

#----- Public Package -----#
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F

#----- Module -----#
#None

def data_info(path="/path/file.avi"):
    
    cap = cv2.VideoCapture(os.path.join(path))

    if not cap.isOpened():
        # 正常に読み込めたのかチェックする
        # 読み込めたらTrue、失敗ならFalse
        print("動画の読み込み失敗")
        sys.exit()

    last_image = None
    length = 0

    while True:
        # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
        is_image, frame_img = cap.read()
        if is_image:
            # 画像を保存
            last_image = frame_img
            length += 1
        else:
            # フレーム画像が読込なかったら終了
            break
    
    return length, last_image, dataloader(path)



def dataloader(path="/path/file.avi"):
    
    cap = cv2.VideoCapture(os.path.join(path))

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
            image = torch.from_numpy(frame_img.transpose(2, 0, 1)[[2, 1, 0]]) / 255
            
            #画像サイズが32で割り切れるように設定
            pad_H = (image.shape[1] % 16)
            pad_W = (image.shape[2] % 16)
            if pad_H != 0:
                pad_H = 16 - pad_H
            if pad_W != 0:
                pad_W = 16 - pad_W
            
            image = F.pad(image[None], padding=[0, 0, pad_W, pad_H])[0]
            images.append(image)
        else:
            # フレーム画像が読込なかったら終了
            break

        if len(images) == 16:
            img = torch.stack(images, dim=1)
            images = []
            yield img[None]
    
    append_num = 16 - len(images)
    images = images + [images[-1] for _ in range(append_num)]

    img = torch.stack(images, dim=1)
    yield img[None]


