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
import matplotlib.pyplot as plt
from tqdm import tqdm
#----- Module -----#


def im_read(path, GRAY=False):
    #HW = 4592, 3448
    # 4592 = 7*41*16
    # 3448 = 431 * 8
    img = cv2.imread(path)
    if GRAY:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return torch.from_numpy(img).float() / 255

image_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/dataset/konmo/P1270204", "*")))
image_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/dataset/konmo/data/images", "*")))

inputs1 = im_read(image_paths[-1])



img = cv2.imread(image_paths[-1])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("threshold.png", img_gray)


MAX_CORNERS = 100
BLOCK_SIZE = 3
QUALITY_LEVEL = 0.2
MIN_DISTANCE  = 5
corners = cv2.goodFeaturesToTrack(img_gray,MAX_CORNERS,QUALITY_LEVEL,\
    MIN_DISTANCE,blockSize=BLOCK_SIZE,useHarrisDetector=False)
corners = np.int0(corners)

#検出したコーナー位置に円を描画
for i in corners:
    x,y=i.ravel()
    cv2.circle(img, (x, y), 6, (0, 0, 255), 2)

cv2.imwrite("corner.png", img)

torchvision.utils.save_image(inputs1[None],
                             "data.png",
                             normalize=True)

# make_seg(image_paths)
