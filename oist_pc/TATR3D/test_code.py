#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import time
import copy
import sys
#----- 専用ライブラリ -----#
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import random
import numbers
import numpy as np
import cv2
#----- 自作ライブラリ -----#
from utils.dataset import PTC_Loader, collate_delete_PAD
import utils.utils as ut



def Visualization_Tracking():
    video = []
    for batch_idx, (inputs, targets, point) in enumerate(plot_loader):
        #channelとlengthを入れ替え、batchとlengthをまとめる
        video.append(inputs.transpose(1, 2).flatten(0, 1))

    video = torch.cat(video, dim=0)
    video = video.detach().numpy().copy()
    video = (video*255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(
        'VESICLE100.mp4', fmt, 30., (512, 512))

    for image in video:
        writer.write(image)

    #ファイルを閉じる
    writer.release()


############## dataloader関数##############
def dataload():
    ### data augmentation + preprocceing ###
    
    plot_transform = ut.Compose([])
    
    plot_dataset = PTC_Loader(Molecule="VESICLE", Density="Low", mode="test", length=16, transform=plot_transform)
    
    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    plot_loader = torch.utils.data.DataLoader(
        plot_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=4,
        shuffle=False,
        drop_last=False)
    
    return plot_loader

plot_loader = dataload()
Visualization_Tracking()
