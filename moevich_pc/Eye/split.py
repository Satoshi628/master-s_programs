#coding: utf-8
#----- 標準ライブラリ -----#
import shutil
import os

#----- 専用ライブラリ -----#
import numpy as np
import glob

#----- 自作モジュール -----#
# None

image_list = sorted(glob.glob("training/images/*"))
anno_list = sorted(glob.glob("training/1st_manual/*"))
mask_list = sorted(glob.glob("training/mask/*"))

for idx in range(len(image_list)):
    print(image_list[idx])
    print(anno_list[idx])
    print(mask_list[idx])


# 4分割
image_np = np.array(image_list, dtype=str)
anno_np = np.array(anno_list, dtype=str)
mask_np = np.array(mask_list, dtype=str)

image_np = np.split(image_np, 4)
anno_np = np.split(anno_np, 4)
mask_np = np.split(mask_np, 4)

image_list = [img.tolist() for img in image_np]
anno_list = [anno.tolist() for anno in anno_np]
mask_list = [mask.tolist() for mask in mask_np]

# split作成
split = {"train_1": [0, 1, 2],
        "test_1": [3],
        "train_2": [1, 2, 3],
        "test_2": [0],
        "train_3": [2, 3, 0],
        "test_3": [1],
        "train_4": [3, 0, 1],
        "test_4": [2]
        }

# dataコピー
for split_name, split in split.items():
    if not os.path.exists(split_name):
        os.mkdir(split_name)
    if not os.path.exists(split_name + "/1st_manual"):
        os.mkdir(split_name + "/1st_manual")
    if not os.path.exists(split_name + "/images"):
        os.mkdir(split_name + "/images")
    if not os.path.exists(split_name + "/mask"):
        os.mkdir(split_name + "/mask")
    split_anno = [item for idx in [0, 1, 2] for item in anno_list[idx]]
    split_image = [item for idx in [0, 1, 2] for item in image_list[idx]]
    split_mask = [item for idx in [0, 1, 2] for item in mask_list[idx]]
    
    for anno in split_anno:
        shutil.copy(anno, os.path.join(split_name, "1st_manual", anno.split("/")[-1]))
    for image in split_image:
        shutil.copy(image, os.path.join(split_name, "images", image.split("/")[-1]))
    for mask in split_mask:
        shutil.copy(mask, os.path.join(split_name, "mask", mask.split("/")[-1]))
    



#shutil.copy(image_list[0], "test.tif")