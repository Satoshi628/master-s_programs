# -*- coding: utf-8 -*-
import os
import re

from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
from nsfw_detector import predict

PORN_TH = 0.5

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def predict_NSFW(model, img_dir):
    image_paths = sorted(glob(os.path.join(img_dir, "*")))
    if len(image_paths) == 0:
        return
    images = []
    for path in image_paths:
        img = imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = img/255.
        images.append(img)
    
    images = np.stack(images)
    output = predict.classify_nd(model, images)
    for out_dict, path in zip(output, image_paths):
        if not out_dict["porn"] < PORN_TH:
            os.remove(path)
            print(path)

if __name__ == "__main__":
    clean_path = "C:\\Users\\hotta_mini\\affi_data\\images_MGS"
    model = predict.load_model('C:\\Users\\hotta_mini\\affi_data\\NSFW\\nsfw.299x299.h5')

    img_paths = glob(os.path.join(clean_path, "*"))

    for img_dir in tqdm(img_paths, leave=False):

        #画像採取
        predict_NSFW(model, img_dir)





