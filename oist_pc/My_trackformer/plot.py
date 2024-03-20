#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import sys
sys.path.append('../')
#----- 専用ライブラリ -----#
import sacred
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import yaml
import cv2

#----- 自作ライブラリ -----#
from dataloader import costom_loader
from util.misc import nested_dict_to_namespace

#mean,std
#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def plot():
    obj_detect_checkpoint_file = "custom_dataset_train_deformable"
    obj_detect_config_path = os.path.join(
        obj_detect_checkpoint_file,
        'config.yaml')
    obj_detect_args = nested_dict_to_namespace(
        yaml.unsafe_load(open(obj_detect_config_path)))
    img_transform = obj_detect_args.img_transform

    data_loader_val = DataLoader(torch.utils.data.Subset(costom_loader("VESICLE_1", img_transform), range(100)))

    down_samp = nn.Upsample((512, 512))
    

    #track_id,frame,x1,y2,x2,y2
    track = np.load("track.npy")

    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('trackformer_track.mp4', fmt, 30., (512, 512))

    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])

    for frame, frame_data in enumerate(data_loader_val):
        image = down_samp(frame_data['img'])[0].detach().numpy().copy()
        image = image * std[:, None, None] + mean[:, None, None]
        image = (image * 255).astype(np.uint8).transpose(1, 2, 0)

        frame_detects = track[track[:, 1] == frame].astype(np.int16)

        for detect in frame_detects:
            image = cv2.rectangle(image.copy(),
                                    pt1=detect[[2, 3]].tolist(),
                                    pt2=detect[[4, 5]].tolist(),
                                    color=(0, 0, 255),
                                    thickness=1,
                                    lineType=cv2.LINE_8)
            
            center_X = int((detect[2] + detect[4]) * 0.5)
            center_Y = int((detect[3] + detect[5]) * 0.5)

            image = cv2.putText(image,
                                text=detect[0].astype(np.str_),
                                org=(center_X - 3, center_Y),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=(255, 255, 255),
                                thickness=1)
        writer.write(image)

    #ファイルを閉じる
    writer.release()


if __name__ == '__main__':
    plot()
    pass
