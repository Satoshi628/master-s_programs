
#coding: utf-8
#----- 標準ライブラリ -----#
import os

#----- 専用ライブラリ -----#
import cv2
import numpy as np
import matplotlib.pyplot as plt

#----- 自作ライブラリ -----#
# None


def save_tracks(path, tracks):
    path = os.path.join(path, "tracks")
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    track_ids = np.unique(tracks[:, 1])
    for tid in track_ids:
        track = tracks[tracks[:, 1] == tid]
        np.savetxt(os.path.join(path, f"track_{int(tid):04}.txt"), track, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])

def save_track_video(path, video, frame=10):
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    WH = video[0].shape[:2][::-1]
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(os.path.join(path, 'track_video.mp4'), fmt, frame, WH)

    for image in video:
        writer.write(image[:,:,[2,1,0]])

    #ファイルを閉じる
    writer.release()
