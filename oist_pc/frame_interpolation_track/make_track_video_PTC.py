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
import matplotlib.pyplot as plt

#----- 自作ライブラリ -----#
from utils.dataset import PTC_Loader
import utils.transforms as tf


PAD_ID = -100


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Visualization_Tracking(cfg):
    video = PTC_Loader(root_dir=cfg.OIST.root_dir, Molecule=cfg.OIST.Staining, Density="Low", mode="test", split=cfg.OIST.split).get_videos()

    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    tracks = np.loadtxt("result/track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}
    print(tracks.shape)
    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('result/track.mp4', fmt, 30., (512, 512))
    writer_notrack = cv2.VideoWriter('result/video.mp4', fmt, 30., (512, 512))

    for idx, image in enumerate(video):
        image = image.copy()
        writer_notrack.write(image)

        track = tracks[tracks[:, 0] == idx]
        for coord in track:
            use_color = random_color[coord[1]]
            image = cv2.circle(image,
                                center=(int(coord[2]), int(coord[3])),
                                radius=6,
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

def EP_result():
    loader = OIST_Loader(Staining="GFP-Mid", mode="test", split=0, length=4)
    video = loader.get_videos()
    EP_imgs = loader.get_EP_images()

    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
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


def plot_histogram(cfg):
    label_tracks = OIST_Loader_Train30_2(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="test", split=cfg.OIST.split, length=4).get_track()
    #[video len, det num, [frame, id, x, y]] => [video len* det num, [frame, id, x, y]]
    label_tracks = label_tracks.flatten(0, 1)
    label_tracks = label_tracks[label_tracks[:, 0] != PAD_ID].numpy()
    
    predict_tracks = np.loadtxt("result/track.txt", delimiter=',').astype(np.int32)
    
    _, label_length = np.unique(label_tracks[:, 1], return_counts=True)
    _, predict_length = np.unique(predict_tracks[:, 1], return_counts=True)
    
    
    plt.clf()
    plt.hist(label_length, bins=50)
    plt.savefig("tracklen_label_histogram.png")

    plt.clf()
    plt.hist(predict_length, bins=50)
    plt.savefig("tracklen_predict_histogram.png")


############## main ##############
@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    #Trackingの可視化
    Visualization_Tracking(cfg)
    #EP_result()

    # track length histgram
    # plot_histogram(cfg)



if __name__ == '__main__':
    main()
