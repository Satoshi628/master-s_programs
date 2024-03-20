#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
import math

#----- Public Package -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import cv2

#----- Module -----#
from utils.dataset import OIST_Loader, OIST_Loader_Test30FPS, collate_delete_PAD
import utils.transforms as tf
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_detection_copy, MPM_to_Track_multi
from utils.evaluation import Object_Detection, Object_Tracking
from models.Unet_3D import UNet_3D

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([])
    
    test_dataset = OIST_Loader_Test30FPS(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="test", split=cfg.OIST.split, length=16, delay=2, transform=test_transform)
    #move_speed_max = OIST_Loader_Test30FPS(Staining="GFP-Mid", mode="train", split=0, length=16).get_max_move_speed()
    #move_speed_max = [4.756889343261719, 5.734686374664307]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader


def Visualization_Tracking():
    video = np.load("/mnt/kamiya/dataset/OIST_new/Sim_immobile_Density_mov_GFP/Sample1/EPmaps.npz")['arr_0']

    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)
    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('15fps_video.mp4', fmt, 15., (512, 512))

    for idx, image in enumerate(video):
        image = image.copy()
        writer.write(image)

    #ファイルを閉じる
    writer.release()

def main():
    Visualization_Tracking()



############## main ##############
if __name__ == '__main__':
    main()
