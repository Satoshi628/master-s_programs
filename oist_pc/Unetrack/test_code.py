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

#----- Module -----#
from utils.dataset import OISTLoader, collate_delete_PAD
from utils.guided_backprop import GuidedBackprop
import utils.transforms as tf
from models.Unet_3D import UNet_3D, UNet_3D_FA


transform = tf.Compose([])
dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov",
                        mode="test",
                        Staning="GFP",
                        split=0,
                        length=16,
                        transform=transform)

loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        drop_last=False)


device = torch.device('cuda:0')
model = UNet_3D(in_channels=1, n_classes=1, noise_strength=0.4).cuda(device)
model_path = "/mnt/kamiya/code/Unetrack/outputs/no_assign/result/model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

GBP = GuidedBackprop(model)
# モデル→推論モード

for batch_idx, (inputs, targets, point) in enumerate(tqdm(loader, leave=False)):
    #input.size() => [batch,channel,length,H,W]
    #targets.size() => [batch,length,H,W]
    #point.size() => [batch,length,num,4(frame,id,x,y)]
    #GPUモード高速化。入力、labelはfloat32
    inputs = inputs.cuda(device, non_blocking=True)
    point = point.cuda(device, non_blocking=True)
    point = point.long()
    grad = GBP.generate_gradients(inputs, point[0, 7, 0, 2:])
    grad = grad.max(dim=0)[0]
    print(grad.shape)
    torchvision.utils.save_image(grad[:, None],
                            "test.png",
                            normalize=True)
    input("owa")