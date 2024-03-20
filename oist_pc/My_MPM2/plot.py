#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import time
from functools import lru_cache
#----- 専用ライブラリ -----#
import cv2
from pylab import *
import h5py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import random
import numbers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#----- 自作ライブラリ -----#
from utils.dataset import C2C12Loader, OISTLoader
import utils.utils_C2C12 as uc
import utils.utils_OIST as uo

def Visualization_Tracking():
    video = []
    
    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for inputs, _ in plot_loader:
        #channelとlengthを入れ替え、batchとlengthをまとめる
        video.append(inputs[:, :1])

    video.append(inputs[:, 1:])

    video = torch.cat(video, dim=0)
    video = video.detach().numpy().copy()
    video = (video * 255).astype(np.uint8)
    #video[100,1,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)

    #trackingデータロード
    #[num,4(frame,id,x,y)]
    track = np.load("/mnt/kamiya/code/My_MPM2/outputs/GFP/0/result/track.npy").astype(np.int16)

    track_ids = np.unique(track[:, 1])

    track = [track[(track[:, 0] == frame) & (track[:, 2] >= 0)]
            for frame in np.unique(track[:, 0])]

    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(
        'result/video/track.mp4', fmt, 30., (512, 512))

    random_color = {track_id.astype(np.str_): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    for image, pre in zip(video, track):
        image = image.copy()
        for coord in pre:
            use_color = random_color[coord[1].astype(np.str_)]
            image = cv2.circle(image,
                                center=(coord[2], coord[3]),
                                radius=6,
                                color=use_color,
                                thickness=1)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(coord[2], coord[3]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=use_color,
                                thickness=1)

        writer.write(image)

    #ファイルを閉じる
    writer.release()


############## dataloader関数##############
def dataload():
    test_transform = uo.Compose([
                                #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])
        
    test_dataset = OISTLoader(Density="Sim",
                                mode="test",
                                use_iter=[1],
                                transform=test_transform)
    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, drop_last=False)
    
    return test_loader


############## main ##############
if __name__ == '__main__':

    if not os.path.exists("result"):
        os.mkdir("result")
    
    if not os.path.exists("result/video"):
        os.mkdir("result/video")

    plot_loader = dataload()

    #Trackingの可視化
    Visualization_Tracking()
