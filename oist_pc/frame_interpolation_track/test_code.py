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
import lap
#----- Module -----#
from models.Unet_3D import UNet_3D_multi
from utils.dataset import OIST_Loader_Train30_2, collate_delete_PAD
import utils.transforms as tf

device = "cuda:1"


val_transform = tf.Compose([])
val_dataset = OIST_Loader_Train30_2(root_dir="/mnt/kamiya/dataset/OIST2", Staining="GFP", mode="val", split=0, length=16, delay=1, transform=val_transform)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    collate_fn=collate_delete_PAD,
    batch_size=1,
    shuffle=False,
    drop_last=False)

model = UNet_3D_multi(in_channels=1,
                n_classes=3,
                noise_strength=0.4,
                delay_upsample=0).cuda(device)

model_path = "outputs/GFP_new_3D_2class/0/result/model.pth"
model.load_state_dict(torch.load(model_path))

with torch.no_grad():
    for data in val_loader:
        inputs = data[0].cuda(device, non_blocking=True)
        output, coord = model.get_coord(inputs)
        output = output[0,:,:16].transpose(0,1)
        torchvision.utils.save_image(output, "multi_map.png")
        torchvision.utils.save_image(output[:,:1], "multi_map1.png")
        torchvision.utils.save_image(output[:,1:2], "multi_map2.png")
        torchvision.utils.save_image(output[:,2:], "multi_map3.png")
        print(output.shape)
        input()