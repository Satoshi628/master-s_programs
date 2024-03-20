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
from utils.dataset import OIST_Loader_Test30FPS
from utils.evaluation import Object_Tracking
from utils.MPM_to_track import global_association_solver

def test(cfg):
    label_tracks = OIST_Loader_Test30FPS(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="test", split=cfg.OIST.split, length=16).CP_data
    predict_tracks = np.loadtxt(f"result/track.txt", delimiter=',').astype(np.float32)
    predict_tracks = torch.from_numpy(predict_tracks)

    Tracking_Eva = Object_Tracking()
    Tracking_Eva.update(predict_tracks, label_tracks)
    acc = Tracking_Eva()
    print("preprocess Evalution")
    for k, v in acc.items():
        print(f"{k}".ljust(20) + f":{v}")

    track = global_association_solver(predict_tracks)
    print(predict_tracks.shape)
    print(track.shape)
    
    Tracking_Eva = Object_Tracking()
    Tracking_Eva.update(track, label_tracks)
    acc = Tracking_Eva()

    print("preprocess Evalution")
    for k, v in acc.items():
        print(f"{k}".ljust(20) + f":{v}")

    
    video =  OIST_Loader_Test30FPS(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="test", split=cfg.OIST.split, length=4).get_videos()

    video = np.stack(video, axis=0)
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    tracks = track.numpy()
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('result/track_global_algo.mp4', fmt, 30., (512, 512))

    for idx, image in enumerate(video):
        image = image.copy()

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


############## main ##############
@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    test(cfg)


if __name__ == '__main__':
    main()
