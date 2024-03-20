#coding: utf-8
#----- Standard Library -----#
import os
import sys

#----- Public Package -----#
import glob
import numpy as np
import torch
import torchvision
import cv2
from tqdm import tqdm

#----- Module -----#
#None

def video2images(video_path, save_dir):

    print("save path".ljust(20) + f":{save_dir}")
    print("video path".ljust(20) + f":{video_path}")
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        # 正常に読み込めたのかチェックする
        # 読み込めたらTrue、失敗ならFalse
        print("動画の読み込み失敗")
        sys.exit()
    
    bar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digit = 6
    n = 0
    while True:
        # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
        is_image, frame_img = cap.read()
        bar.update()
        if is_image:
            cv2.imwrite(os.path.join(save_dir, str(n).zfill(digit) + ".png"), frame_img)
            n += 1
        else:
            # フレーム画像が読込なかったら終了
            break
    bar.close()

def main():
    video_paths = sorted(glob.glob("videos/*/*.avi"))
    directory_dirs = [os.path.join(os.path.dirname(file_path), "images") for file_path in video_paths]
    for video_path, save_dir in zip(video_paths, directory_dirs):
        video2images(video_path, save_dir)


############## main ##############
if __name__ == '__main__':
    main()
