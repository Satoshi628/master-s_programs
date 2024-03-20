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
from utils.dataset import Zenigoke_Loader_Exp, Prompt_Loader


PAD_ID = -100


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Visualization_Tracking(cfg):
    video =  Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name).get_videos()

    video = np.stack(video, axis=0)
    video = video.astype(np.uint8)
    HW = video.shape[1:3]
    print(video.shape)
    video_name = cfg.dataset.video_name.split(".")[0]
    #[time, id, x1,y1,x2,y2, score]
    tracks = np.loadtxt(f"result/{video_name}-track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('result/track.mp4', fmt, 30., HW)
    writer_notrack = cv2.VideoWriter('result/video.mp4', fmt, 30., HW)

    for idx, image in enumerate(video):
        image = image.copy()
        writer_notrack.write(image)

        track = tracks[tracks[:, 0] == idx]
        for coord in track:
            use_color = random_color[coord[1]]
            bbox = coord[-5:-1]
            bbox[2:] -= bbox[:2]
            image = cv2.rectangle(image, list(bbox.astype(np.int32)), color=use_color, thickness=4)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(int(bbox[0]), int(bbox[1])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.,
                                color=use_color,
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()
    writer_notrack.release()


def Visualization_Detection(cfg):
    video =  Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name).get_videos()

    video = np.stack(video, axis=0)
    video = video.astype(np.uint8)
    HW = video.shape[1:3]
    print(video.shape)
    video_name = cfg.dataset.video_name.split(".")[0]
    dets = np.loadtxt(f"result/{video_name}-dets.txt", delimiter=',')
    
    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('result/dets.mp4', fmt, 30., HW)

    for idx, image in enumerate(video):
        image = image.copy()

        dets_per_image = dets[dets[:, 0] == idx]
        for bbox in dets_per_image:
            box = bbox[1:5]
            box[-2:] -= box[:2]
            image = cv2.rectangle(image, list(box.astype(np.int32)), color=(0, 0, 255), thickness=4)
            image = cv2.putText(image,
                                    text=f"{bbox[-1]:.2%}",
                                    org=(int(box[0]), int(box[1])),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1.,
                                    color=(0, 0, 255),
                                    thickness=2)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()


############## main ##############
@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    #Trackingの可視化
    Visualization_Tracking(cfg)
    Visualization_Detection(cfg)


if __name__ == '__main__':
    main()
