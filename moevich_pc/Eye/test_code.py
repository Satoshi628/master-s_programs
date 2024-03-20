#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
from time import sleep

#----- External Library -----#
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import (Compose,
    Resize,
    RandomCrop,
    RandomHorizontalFlip,
    Pad,
    RandomVerticalFlip)
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm
import cv2

#----- My Module -----#
#None


path = "/mnt/hdd1/kamiya/dataset/Eye_edit/HE_mask/case0093.png"
mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
retval, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
stats = stats[1:]
flag = (stats[:, -1] < 10) | (stats[:,2] * stats[:,3] > 1000) | (stats[:,2] / stats[:,3] > 3.) | (stats[:,2] / stats[:,3] < 0.3)

print(retval - 1 - flag.sum())
print(stats[~flag, -1].sum())
print(stats[~flag, -1])
