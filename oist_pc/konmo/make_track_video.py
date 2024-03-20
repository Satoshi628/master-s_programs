#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import time
#----- 専用ライブラリ -----#
import glob
import hydra
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F

#----- 自作ライブラリ -----#
from utils.dataset import Konmo_Video_Loader


frame = 5.
resize = 2.

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Visualization_Tracking(cfg, dataloader):
    video = dataloader.images
    WH = [int(wh // resize) for wh in video[0].shape[1:][::-1]]

    #[time, id, x,y]

    tracks = np.loadtxt(f"result/track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('result/track.mp4', fmt, frame, WH)

    for idx, image in enumerate(video):
        image = cv2.resize(image.transpose(1,2,0),None, fx=1/resize, fy=1/resize)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        track = tracks[tracks[:, 0] == idx]
        for coord in track:
            use_color = random_color[coord[1]]
            point = coord[2:4] // resize
            img = cv2.circle(image,
                        center=list(point.astype(np.int32)),
                        radius=3,
                        color=use_color,
                        thickness=2)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(int(point[0]), int(point[1])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.,
                                color=use_color,
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()

def crop(img, center_xy, WH):
    img = torch.from_numpy(img.transpose(2, 0, 1))
    print(img.shape)
    print(center_xy)
    print(WH)

    img = F.crop(img, center_xy[1] - WH[1] // 2, center_xy[0] - WH[0] // 2, WH[1], WH[0])
    return img.numpy().transpose(1, 2, 0)

def Visualization_Tracking_one(cfg, dataloader):
    video = dataloader.images
    
    WH = [wh / resize for wh in video[0].shape[:2][::-1]]
    print(video.shape)
    tracks = np.loadtxt(f"result/track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter('result/track-one.mp4', fmt, frame, WH)

    search_id = 5
    center_xy = np.zeros([2],dtype=int)

    for idx, image in enumerate(video):
        image = cv2.resize(image.transpose(1,2,0), None, fx=1/resize, fy=1/resize)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        track = tracks[tracks[:, 0] == idx]
        if track[track[:, 1] == search_id].shape[0] == 1:
            center_xy = track[track[:, 1] == search_id][0, 2:4].astype(np.int32)
        
        for coord in track:
            use_color = random_color[coord[1]]
            bbox = coord[-5:-1]
            bbox[2:] -= bbox[:2]
            image = cv2.rectangle(image, list(bbox.astype(np.int32)), color=use_color, thickness=2)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(int(bbox[0]), int(bbox[1])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.,
                                color=use_color,
                                thickness=1)
        
        image = crop(image, center_xy, WH)
        writer.write(image)

    #ファイルを閉じる
    writer.release()


def Visualization_Trajectory(cfg, crop):
    
    path = sorted(glob.glob(os.path.join(cfg.dataset.root_dir, "P1270204", "*")))[-1]


    img = cv2.imread(path)[crop[1]:crop[3], crop[0]:crop[2]]

    img = cv2.resize(img,None, fx=1/resize, fy=1/resize)

    print(img.shape)
    tracks = np.loadtxt(f"result/track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(0, 128) for _ in range(3)] for track_id in track_ids}


    for tid in track_ids:
        track = tracks[tracks[:, 1] == tid]
        use_color = random_color[tid]
        point = (track[:, 2:4] // resize).astype(np.int32)


        past_point = None
        for p in point:
            # img = cv2.circle(img,
            #             center=(int(p[0]), int(p[1])),
            #             radius=3,
            #             color=use_color,
            #             thickness=-1)
            if past_point is not None:
                cv2.line(img, past_point, p, use_color, thickness=2, lineType=cv2.LINE_AA)
            past_point = p

    cv2.imwrite('result/trajectory.png', img)



############## main ##############
@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    crop = [1500, 0, 2198, 4592]
    Visualization_Trajectory(cfg, crop)
    # dataloader = Konmo_Video_Loader(root_dir=cfg.dataset.root_dir, crop=[1500, 0, 2198, 4592], data_num=None, transform=None)
    #Trackingの可視化
    # Visualization_Tracking(cfg, dataloader)
    # Visualization_Tracking_one(cfg, dataloader)



if __name__ == '__main__':
    main()
