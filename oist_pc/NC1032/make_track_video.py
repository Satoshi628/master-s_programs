#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import time
#----- 専用ライブラリ -----#
import hydra
import cv2
import numpy as np

#----- 自作ライブラリ -----#
from utils.dataset import OIST_Loader, OIST_Loader_Test, collate_delete_PAD
import utils.transforms as tf

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Visualization_Tracking():
    video = OIST_Loader_Test(length=16).get_videos()
    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = video[:, None]
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    tracks = np.loadtxt("result/track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    im_size = video.shape[-3:-1][::-1]
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('result/track.mp4', fmt, 10., im_size)
    writer_notrack = cv2.VideoWriter('result/video.mp4', fmt, 10., im_size)

    for idx, image in enumerate(video):
        image = image.copy()
        writer_notrack.write(image)

        track = tracks[tracks[:, 0] == idx]
        for coord in track:
            use_color = random_color[coord[1]]
            image = cv2.circle(image,
                                center=(int(coord[2]), int(coord[3])),
                                radius=4,
                                color=use_color,
                                thickness=1)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(int(coord[2]), int(coord[3])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=use_color,
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()
    writer_notrack.release()


def Visualization_GT():
    video = OIST_Loader_Test(length=16).get_videos()
    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = video[:, None]
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    tracks = np.loadtxt("/mnt/kamiya/dataset/NC1032/det.txt", delimiter=',').astype(np.int32)
    
    
    #labelのdetectionでのTrackingを動画化
    im_size = video.shape[-3:-1][::-1]
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('result/GT.mp4', fmt, 10., im_size)

    for idx, image in enumerate(video):
        image = image.copy()
        track = tracks[tracks[:, 0] == idx + 1]
        for coord in track:
            image = cv2.circle(image,
                                center=(int(coord[1]), int(coord[2])),
                                radius=4,
                                color=[0,0,255],
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()


def EP_result():
    loader = OIST_Loader_Test(length=16)
    video = loader.get_videos()
    EP_imgs = loader.get_EP_images()

    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = video[:, None]
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    EP_imgs = np.stack(EP_imgs, axis=0)
    EP_imgs = (EP_imgs * 255).astype(np.uint8)
    #EP_imgs[100,512,512]から[100,512,512,3]にする
    EP_imgs = np.tile(EP_imgs, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(EP_imgs.shape)

    mkdir("imgs")
    mkdir("EPs")
    for idx, (img, EP_img) in enumerate(zip(video, EP_imgs)):
        cv2.imwrite(f"imgs/img{idx:06}.png", img)
        cv2.imwrite(f"EPs/EP{idx:06}.png", EP_img)


############## main ##############
@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    #Trackingの可視化
    Visualization_Tracking()
    Visualization_GT()
    #EP_result()


if __name__ == '__main__':
    main()
