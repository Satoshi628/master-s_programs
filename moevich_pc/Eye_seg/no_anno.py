#coding: utf-8
#----- 標準ライブラリ -----#
import os
import re

#----- 専用ライブラリ -----#
import glob
import numpy as np
from PIL import Image
import cv2
#----- 自作モジュール -----#
#None

def main():
    path = "/mnt/hdd1/kamiya/dataset/IDRID/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates"
    path = "/mnt/hdd1/kamiya/dataset/IDRID/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages"
    #path = "/mnt/hdd1/kamiya/dataset/IDRID/2. All Segmentation Groundtruths/b. Testing Set/4. Soft Exudates"

    clean_image = np.zeros([2848, 4288, 3],dtype=np.uint8)
    #cv2.circle(clean_image, (125, 420), 71, (255, 255, 255), thickness=-1)
    files = glob.glob(path + "/*")
    exist_number = [int(re.search(r"IDRiD_\d+",f).group()[-2:]) for f in files]
    no_exist_number = [num for num in range(1,55) if num not in exist_number]
    #no_exist_number = [num for num in range(55,82) if num not in exist_number]

    print(no_exist_number)
    print(len(no_exist_number))
    for num in no_exist_number:
        cv2.imwrite(f"{path}/IDRiD_{num:02}_HE.png", clean_image)

    #images = torch.nonzero(images)
    #print(images.shape)
    #image_idx, px_count = torch.unique(images[:, 0], return_counts=True)
    #print(image_idx)
    #print(px_count)







############## main ##############
if __name__ == '__main__':
    main()
