#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from tqdm import tqdm

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import apex
from apex import amp, optimizers
import torch.nn as nn
import torchvision.transforms as tf

from PIL import Image

image_name = "/mnt/kamiya/dataset/SIGNATE_TECHNOPRO/train/0000.png"
# image_name = "/mnt/kamiya/dataset/SIGNATE_TECHNOPRO/train/0001.png"
img = Image.open(image_name).convert("RGB")
img = tf.ToTensor()(img)

fft = torch.fft.fft2(img)
fft = torch.fft.fftshift(fft, dim=[-1,-2])

mask = torch.ones(fft.shape)
x = torch.linspace(-1,1,mask.shape[-1])
y = torch.linspace(-1,1,mask.shape[-2])
xy = torch.sqrt(x[None]**2 + y[:, None]**2)
mask = xy / xy.max()
print(torch.topk(fft.abs() * mask**2.,).mean())
fft = torch.fft.fftshift(fft, dim=[-1,-2])

new_img = torch.fft.ifft2(fft).abs()
save = torch.stack([img, new_img])
torchvision.utils.save_image(save, "fft.png")
