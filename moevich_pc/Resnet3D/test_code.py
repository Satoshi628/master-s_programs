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

#----- My Module -----#
from utils.utils import CT_ndarrayToTensor
from utils.dataset import PVT_Loader
from utils.visualization import CAM
from models.resnet3D import resnet3D50


inputs = torch.rand([8,2])
targets = torch.randint(2,[8])
print(targets)
print(inputs)
print(torch.gather(inputs,-1,targets[:,None]))