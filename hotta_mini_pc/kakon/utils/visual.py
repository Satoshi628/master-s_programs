
#coding: utf-8
#----- 標準ライブラリ -----#
import random

#----- 専用ライブラリ -----#
import cv2
import numpy as np

#----- 自作ライブラリ -----#
# None

def VisualizationTrajectory(img, tracks):
    img = img.astype(np.uint8)
    tracks = tracks.astype(np.int32)

    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(0, 128) for _ in range(3)] for track_id in track_ids}


    for tid in track_ids:
        track = tracks[tracks[:, 1] == tid]
        use_color = random_color[tid]
        point = track[:, 2:4]

        past_point = None
        for p in point:
            if past_point is not None:
                cv2.arrowedLine(img, past_point, p, use_color, thickness=2, tipLength=0.4)
            past_point = p

    return img