#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
import math

#----- Public Package -----#
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn
from tqdm import tqdm
#----- Module -----#

def im_read(path, GRAY=True):
    img = cv2.imread(path)
    return img


image_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/dataset/root/data/sample1/video", "*")))
track_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/dataset/root/data/sample1/tracks", "*")))

img = im_read(image_paths[-1])
H, W, _ = img.shape

color_list = [[random.randint(64, 256), random.randint(64, 256), random.randint(64, 256)] for _ in range(len(track_paths))]

movement = []
coordinate = []

for idx, track_path in enumerate(track_paths):
    color = color_list[idx]
    track = np.loadtxt(track_path, delimiter=',').astype(np.int32)
    track[:, 2] = H - track[:, 2]

    movement.append(np.diff(track[:, 1:], n=1, axis=0))
    coordinate.append(track[:, 1:] - track[:, 1:].mean(axis=0))

    past_coord = None
    for t in track:
        
        img = cv2.circle(img,
                        center=(int(t[1]), int(t[2])),
                        radius=5,
                        color=color,
                        thickness=2)

        if past_coord is not None:
            cv2.line(img,
                pt1=(past_coord[1], past_coord[2]),
                pt2=(t[1], t[2]),
                color=color,
                thickness=3,
                shift=0)
        
        past_coord = t

move_values = []
for mv in movement:
    move_values.append(np.sqrt((mv ** 2).sum(axis=-1)))

move_values = np.concatenate(move_values)
movement = np.concatenate(movement)
coordinate = np.concatenate(coordinate)

print(f"velocity average:{move_values.mean():.2f}")
print(f"velocity std:{np.std(move_values):.2f}")


print(f"x velocity average:{movement[:,0].mean():.4f}")
print(f"x velocity std:{np.std(movement[:,0]):.4f}")
print(f"x velocity variance:{np.std(movement[:,0])**2:.4f}")

print(f"y velocity average:{movement[:,1].mean():.4f}")
print(f"y velocity std:{np.std(movement[:,1]):.4f}")
print(f"y velocity variance:{np.std(movement[:,1])**2:.4f}")


print(f"x coordinate average:{coordinate[:,0].mean():.2f}")
print(f"x coordinate std:{np.std(coordinate[:,0]):.2f}")

print(f"y coordinate average:{coordinate[:,1].mean():.2f}")
print(f"y coordinate std:{np.std(coordinate[:,1]):.2f}")

cv2.imwrite("sample1_track.png", img)