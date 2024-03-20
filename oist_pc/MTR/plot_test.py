#coding: utf-8
#----- Standard Library -----#
import os
import random
import sys

#----- Public Package -----#
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import cv2

#----- Module -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.loss import movement_Loss
from utils.dataset import OISTLoader, OISTLoader_add, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.Transformer_to_track import Transformer_to_Track
from utils.evaluation import Object_Tracking
from models.PTGT import PTGT
from models.MTR import MTR
"""
tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
"""
############## dataloader function ##############
def dataload():
    val_transform = tf.Compose([])
    #val_transform = tf.Compose([tf.CenterCrop(size=(256, 256)),
    #                            tf.RandomHorizontalFlip(p=1.0),
    #                            tf.RandomVerticalFlip(p=1.0)])
    val_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/Sim_immobile_Density_mov", mode="val", data="GFP_im_Low", length=16, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return val_loader

def Visualization_Tracking():
    val_loader = dataload()
    video = []
    track = []
    for batch_idx, (inputs, targets, point) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
        video.append(inputs[0,0])
        track.append(point[0].transpose(0, 1))
    
    video = torch.cat(video, dim=0)
    track = torch.nn.utils.rnn.pad_sequence(track, batch_first=True, padding_value=-1.)
    track = track.transpose(1, 2).flatten(0, 1)
    track = track.detach().numpy().copy()
    track = track[track[:, :, 0, 0] != -1]
    
    video = (video * 255).detach().numpy().copy().astype(np.uint8)
    video = np.tile(np.expand_dims(video, -1), (1, 1, 1, 3))

    track_ids, counts = np.unique(track[:, 0, 1], return_counts=True)
    
    one_track = track[track[:, 0, 1] == 2]

    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('track.mp4', fmt, 30., (512, 512))

    use_color = [0, 0, 255]
    
    coord = one_track[0, 1]
    for image, pre in zip(video, one_track):
        image = image.copy()
        image = cv2.circle(image,
                            center=(round(coord[0]), round(coord[1])),
                            radius=6,
                            color=use_color,
                            thickness=1)
        writer.write(image)
        coord += pre[2]

    #ファイルを閉じる
    writer.release()


def main():
    Visualization_Tracking()

############## main ##############
if __name__ == '__main__':
    main()
