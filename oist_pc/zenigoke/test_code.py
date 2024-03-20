#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
import math

#----- Public Package -----#
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
from tqdm import tqdm
#----- Module -----#


filters = torch.rand([1, 4, 4, 4]).cuda()
inputs = torch.rand([1, 4, 100, 100]).cuda()
padding = [f-2 for f in filters.shape[-2:]]

print(F.conv2d(inputs, filters, padding=padding).shape)
