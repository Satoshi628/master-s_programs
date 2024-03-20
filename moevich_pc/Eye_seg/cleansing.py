#coding: utf-8
#----- 標準ライブラリ -----#
import matplotlib.pyplot as plt
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import glob
import hydra
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as Ft
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
#----- 自作モジュール -----#
#None

def generate_cricle(H,W):
    mask = np.zeros([H, W, 3], dtype=np.uint8)

    H = H.item()
    W = W.item()
    R = min([H,W]) // 2

    cv2.circle(mask, (W//2, H//2), R, (255, 255, 255), thickness=-1)
    mask = mask[:, :, 0].astype(np.float32) / 255
    #正規化
    mask = mask / mask.sum()
    return torch.tensor(mask)[None, None]


@hydra.main(config_path="config", config_name='cleansing.yaml')
def main(cfg):
    
    image_path = sorted(glob.glob("mask/*.png"))
    images = [Ft.to_tensor(Image.open(image_name).convert("L")) for image_name in image_path]
    images = torch.stack(images, dim=0).cuda("cuda:0")

    sample_image = Ft.to_tensor(Image.open("mask/case0001.png").convert("L"))
    idx = torch.nonzero(sample_image[0] == 1)
    H_min = idx[:,0].min().min()
    H_max = idx[:,0].max().max()
    W_min = idx[:,1].min().min()
    W_max = idx[:,1].max().max()
    #奇数にするため
    H_min = H_min - 1
    W_min = W_min - 1
    print(H_max-H_min, W_max-W_min)

    kernel = generate_cricle(H_max-H_min, W_max-W_min)
    kernel = kernel.cuda("cuda:0")
    gaussian_filter = torchvision.transforms.GaussianBlur((H_max-H_min, W_max-W_min), sigma=3)
    maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
    
    cricle_detection = F.conv2d(images, kernel, padding=((H_max-H_min-1) // 2, (W_max-W_min-1) // 2))
    images = (cricle_detection == cricle_detection.flatten(1).max(dim=-1)[0][:, None, None, None]) * 1.0

    for _ in range(2):
        images = gaussian_filter(images)
        images = (images != 0) * (images == maxpool(images)) * 1.0
    

    idx = torch.nonzero(images[:,0])
    clean_image = np.zeros([images.size(0),images.size(2),images.size(3),3],dtype=np.uint8)

    H = (H_max-H_min).item()
    W = (W_max-W_min).item()
    R = min([H,W]) // 2

    for center_p in idx:
        cv2.circle(clean_image[center_p[0]], (center_p[2].item(), center_p[1].item()), R, (255, 255, 255), thickness=-1)

    if not os.path.exists("cleansing_mask"):
        os.mkdir("cleansing_mask")

    for path, save_img in zip(image_path, clean_image):
        file_name = path.split("/")[-1][:-4]
        cv2.imwrite(f"cleansing_mask/{file_name}.png", save_img[:,:,0])


    #images = torch.nonzero(images)
    #print(images.shape)
    #image_idx, px_count = torch.unique(images[:, 0], return_counts=True)
    #print(image_idx)
    #print(px_count)







############## main ##############
if __name__ == '__main__':
    main()
