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
import torch

#----- 自作ライブラリ -----#
from utils.dataset import OIST_Loader_Exp

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Visualization_Tracking(root_dir="/mnt/kamiya/dataset/OIST/Experimental_data/GFP", Staining="CD47-GFP_01.avi"):
    video = OIST_Loader_Exp(root_dir=root_dir, Staining=Staining, length=8).get_videos()
    video = torch.stack(video, axis=0)
    video = video.numpy()
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    video_name = Staining.split(".")[0]
    tracks = np.loadtxt(f"result/{video_name}-track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter(f'result/{video_name}-track.mp4', fmt, 30., (512, 512))
    writer_notrack = cv2.VideoWriter(f'result/{video_name}-video.mp4', fmt, 30., (512, 512))

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
                                fontScale=0.3,
                                color=use_color,
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()
    writer_notrack.release()


############## main ##############
@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    #Trackingの可視化
    Visualization_Tracking(cfg.OIST.root_dir, cfg.OIST.Staining)

    #EP_result()


if __name__ == '__main__':
    main()
