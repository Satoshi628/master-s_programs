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
from utils.dataset import Zenigoke_Loader, Prompt_Loader


PAD_ID = -100
frame = 5.


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def Visualization_Tracking(cfg):
    dataset = Zenigoke_Loader(root_dir=cfg.dataset.root_dir, mode="test", split=cfg.dataset.split, length=16, delay=1, transform=None)
    video = dataset.get_videos()

    video = np.stack(video, axis=0) * 255
    video = video.astype(np.uint8)
    WH = video.shape[1:3][::-1]
    print(video.shape)
    video_name = cfg.dataset.video_name.split(".")[0]
    #[time, id, x1,y1,x2,y2, score]
    tracks = dataset.get_track().flatten(0, 1).numpy()
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, frame, (640, 480))
    writer = cv2.VideoWriter('result/label_track.mp4', fmt, frame, WH)
    writer_notrack = cv2.VideoWriter('result/video.mp4', fmt, frame, WH)

    for idx, image in enumerate(video):
        image = image.copy()
        writer_notrack.write(image)

        track = tracks[tracks[:, 0] == idx]
        for coord in track:
            use_color = random_color[coord[1]]
            center = coord[-2:]
            
            image = cv2.circle(image,
                            center=(int(center[0]), int(center[1])),
                            radius=5,
                            color=use_color,
                            thickness=2)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()
    writer_notrack.release()


def Visualization_Detection(cfg):
    video =  Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name).get_videos()

    video = np.stack(video, axis=0)
    video = video.astype(np.uint8)
    WH = video.shape[1:3]
    print(video.shape)
    video_name = cfg.dataset.video_name.split(".")[0]
    dets = np.loadtxt(f"result/{video_name}-dets.txt", delimiter=',')
    
    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, frame, (640, 480))
    writer = cv2.VideoWriter('result/dets.mp4', fmt, frame, (1346, 1014))

    for idx, image in enumerate(video):
        image = image.copy()

        dets_per_image = dets[dets[:, 0] == idx]
        for bbox in dets_per_image:
            box = bbox[1:5]
            box[-2:] -= box[:2]
            image = cv2.rectangle(image, list(box.astype(np.int32)), color=(0, 0, 255), thickness=2)
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


def Visualization_Trajectory(cfg):
    img =  Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name).get_videos()[-1]
    
    img = img.astype(np.uint8)
    video_name = cfg.dataset.video_name.split(".")[0]
    #[time, id, x1,y1,x2,y2, score]
    tracks = np.loadtxt(f"result/{video_name}-track.txt", delimiter=',').astype(np.int32)
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(0, 128) for _ in range(3)] for track_id in track_ids}


    for tid in track_ids:
        track = tracks[tracks[:, 1] == tid]
        use_color = random_color[tid]
        point = track[:, -5:-1]
        point = np.round(point[:, :2] + 6).astype(np.int32)

        past_point = None
        for p in point:
            # img = cv2.circle(img,
            #             center=(int(p[0]), int(p[1])),
            #             radius=3,
            #             color=use_color,
            #             thickness=-1)
            if past_point is not None:
                cv2.arrowedLine(img, past_point, p, use_color, thickness=2, tipLength=0.4)
            past_point = p

    cv2.imwrite('result/trajectory.png', img)



def Visualization_pre_Trajectory(cfg):
    img =  Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name).get_videos()[-1]
    
    img = img.astype(np.uint8)
    video_name = cfg.dataset.video_name.split(".")[0]
    #[time, id, x1,y1,x2,y2, score]
    tracks = np.load(f"predict_track.npy")
    aaa = 70
    tracks = tracks[aaa:aaa+30]
    print(tracks.shape)
    random_color = {int(idx): [random.randint(0, 128) for _ in range(3)] for idx in range(len(tracks))}


    for idx, track in enumerate(tracks):
        use_color = random_color[idx]
        point = np.round(track).astype(np.int32)

        past_point = None
        for p in point:
            img = cv2.circle(img,
                        center=(int(p[0]), int(p[1])),
                        radius=3,
                        color=use_color,
                        thickness=-1)
            if past_point is not None:
                cv2.line(img, past_point, p, use_color, thickness=2)
            past_point = p

    cv2.imwrite('result/predict_trajectory.png', img)

############## main ##############
@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    #Trackingの可視化
    Visualization_Tracking(cfg)
    # Visualization_Detection(cfg)
    # Visualization_Trajectory(cfg)
    # Visualization_pre_Trajectory(cfg)


if __name__ == '__main__':
    main()
