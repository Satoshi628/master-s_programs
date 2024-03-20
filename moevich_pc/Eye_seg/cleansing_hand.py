#coding: utf-8
#----- 標準ライブラリ -----#
import matplotlib.pyplot as plt
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import glob
import hydra
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as Ft
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
#----- 自作モジュール -----#
#None


@hydra.main(config_path="config", config_name='cleansing.yaml')
def main(cfg):
    file_name = "case20"
    #file_name = "case267"
    # case20  (125, 420)
    # case267 (770, 500)


    clean_image = np.zeros([945, 945, 3],dtype=np.uint8)
    cv2.circle(clean_image, (125, 420), 71, (255, 255, 255), thickness=-1)

    if not os.path.exists("cleansing_mask"):
        os.mkdir("cleansing_mask")

    cv2.imwrite(f"cleansing_mask/{file_name}.png", clean_image[:,:,0])

    #images = torch.nonzero(images)
    #print(images.shape)
    #image_idx, px_count = torch.unique(images[:, 0], return_counts=True)
    #print(image_idx)
    #print(px_count)







############## main ##############
if __name__ == '__main__':
    main()
