#coding: utf-8
#----- Standard Library -----#
import os
import sys
import argparse

#----- Public Package -----#
import numpy as np
import torch
import torchvision
import cv2
from tqdm import tqdm

#----- Module -----#
#None

def main():
    parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

    # 3. parser.add_argumentで受け取る引数を追加していく
    parser.add_argument('-s', '--save_dir', help='save dir', default="video_images")
    parser.add_argument('-r', '--root_dir', help='video root dir')
    parser.add_argument('-n', '--video_name', help='video file name')


    args = parser.parse_args()  # 4. 引数を解析
    video_name = args.video_name.split(".")[0]

    print("save path".ljust(20) + f":{args.save_dir}")
    print("video path".ljust(20) + f":{args.root_dir}" + f"/{args.video_name}")
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    if not os.path.exists(os.path.join(args.save_dir, video_name)):
        os.mkdir(os.path.join(args.save_dir, video_name))
    
    cap = cv2.VideoCapture(os.path.join(args.root_dir, args.video_name))
    
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
            cv2.imwrite(args.save_dir + f"/{video_name}/" + str(n).zfill(digit) + ".png", frame_img)
            n += 1
        else:
            # フレーム画像が読込なかったら終了
            break
    bar.close()




############## main ##############
if __name__ == '__main__':
    main()
