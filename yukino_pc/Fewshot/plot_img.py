#coding: utf-8
#----- 標準ライブラリ -----#
import math

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import cv2

#----- 自作モジュール -----#
#None

label = cv2.imread("/mnt/kamiya/dataset/ACDC/sample_ano.png")
img = torch.from_numpy(label.copy())
temp = img.clone()

color = [[0, 0, 0],
        [189, 220, 254],
        [156, 206, 144],
        [0, 255, 0],
        [0, 255, 255],
        [128, 0, 128],
        [253, 254, 189],
        [255, 0, 0],
        [255, 73, 143]]
color = [[0, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0]]
color = torch.tensor(color)
color = color[:, [2,1,0]]

for idx, c in enumerate(color):
    img[torch.all(c[None,None] == temp,dim=-1)] = idx


onehot = F.one_hot(img[:,:,0].long(), num_classes=9).permute(2,0,1)[None]

edge_range = 2
avgPool = nn.AvgPool2d(2 * edge_range + 1, stride=1, padding=edge_range)

# seg_map is segmentation map [batch,class,H,W] value:0 or 1
# Non differentiable
smooth_map = avgPool(onehot)

# 物体の曲線付近内側までを1.とするフラグを作成
object_edge_inside_flag = onehot * (smooth_map != onehot)

flag = torch.any(object_edge_inside_flag==1, dim=1)
flag = flag.numpy()[0]
label_edge = label.copy()
label_center = label.copy()
label_edge[~flag] = 0
label_center[flag] = 0

cv2.imwrite("/mnt/kamiya/code/Fewshot/label0056_edge.png", label_edge)
cv2.imwrite("/mnt/kamiya/code/Fewshot/label0056_center.png", label_center)
