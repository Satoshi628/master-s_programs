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
import celldetection as cd
#----- Module -----#
pass

# model = cd.models.ResNeXt50UNet(in_channels=3, out_channels=1, nd=2).cuda()

# inputs = torch.rand([1, 3, 256, 256]).cuda()
# outputs = model(inputs)
# print(outputs.shape)
# print(outputs.max())
# print(outputs.min())

EP_data = np.load("/mnt/kamiya/dataset/OIST_new/Sim_immobile_Density_mov_GFP/Sample1/EPmaps.npz")['arr_0'][0]
EP_data = torch.from_numpy(EP_data)
print(EP_data.shape)
torchvision.utils.save_image(EP_data[None],
                            "original.png",
                            normalize=True)

kernelsize = 3
weights = torch.ones([1, 1, kernelsize, kernelsize]).cuda()

softmax_ep = EP_data.cuda() - 0.5
softmax_ep = torch.exp(softmax_ep / 0.07)
softmax_under = F.conv2d(softmax_ep[None], weights, padding=(kernelsize - 1) // 2)
softmax_ep = softmax_ep / (softmax_under + 1)

torchvision.utils.save_image(softmax_ep,
                            "spatial_softmax.png",
                            normalize=True)




