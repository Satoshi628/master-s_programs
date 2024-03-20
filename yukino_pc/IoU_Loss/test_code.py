#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.dataset import Covid19_Loader, Drosophila_Loader, Cityscapse_Loader, Drive_Loader, ACDC_Loader, Synapse_Multi_Organ_Loader



"""
root = "/mnt/kamiya/dataset/covid19"
dataset = Covid19_Loader(root, dataset_type='train')

root = "/mnt/kamiya/dataset/drosophila/data"
dataset = Drosophila_Loader(root, val_area=0, split='train', iteration_number=12 * 4)

root = "/mnt/kamiya/dataset/City_Scapes"
dataset = Cityscapse_Loader(root, dataset_type='val')

root = "/mnt/kamiya/dataset/DRIVE_retina"
dataset = Drive_Loader(root, split="train_1")

root = "/mnt/kamiya/dataset/ACDC"
dataset = ACDC_Loader(root, mode="train", split=1)

root = "/mnt/kamiya/dataset/Synapse_Multi_Organ"
dataset = Synapse_Multi_Organ_Loader(root, mode="train", split=1)
"""