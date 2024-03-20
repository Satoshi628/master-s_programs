#coding: utf-8
#----- Standard Library -----#
import re
import os
import random
import time

#----- Public Package -----#
import cv2
import numpy as np
import torch
import torch.nn as nn
import hydra

#----- Module -----#
from utils.dataset import OISTLoader_plot


def Visualization_Tracking(track_path, video_path):
    video = OISTLoader_plot(data=video_path, frame=100)

    video = video.detach().numpy().copy().astype(np.uint8)
    video = video.transpose(1, 2, 3, 0)

    track = np.load(track_path).astype(np.int32)
    track_ids = np.unique(track[:, 1])

    track = [track[(track[:, 0] == frame) & (track[:, 2] >= 0)]
            for frame in sorted(np.unique(track[:, 0]))]
    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('track.mp4', fmt, 30., (512, 512))


    random_color = {track_id.astype(np.str_): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    for image, pre in zip(video, track):
        image = image.copy()
        for coord in pre:
            use_color = random_color[coord[1].astype(np.str_)]
            image = cv2.circle(image,
                                center=(coord[2], coord[3]),
                                radius=6,
                                color=use_color,
                                thickness=1)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(coord[2], coord[3]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=use_color,
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()


@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    print("analysis data".ljust(20) + f":{cfg.analysis_data}")
    #Trackingの可視化
    Visualization_Tracking("track.npy", video_path=cfg.analysis_data)

############## main ##############
if __name__ == '__main__':
    main()
