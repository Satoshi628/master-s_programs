
#coding: utf-8
#----- 標準ライブラリ -----#
import random

#----- 専用ライブラリ -----#
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#----- 自作ライブラリ -----#
# None


def Visualization_Tracking(images_gene, tracks):
    
    track_ids = np.unique(tracks[:, 1])
    random_color = {int(track_id): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    images = []
    for idx, image in tqdm(enumerate(images_gene), leave=False):
        image = image.copy()

        track = tracks[tracks[:, 0] == idx]
        for coord in track:
            use_color = random_color[coord[1]]
            center = coord[-2:]
            
            image = cv2.circle(image,
                            center=(int(center[0]), int(center[1])),
                            radius=5,
                            color=use_color,
                            thickness=-1)
            image = cv2.putText(image,
                    text=str(int(coord[1])),
                    org=(int(center[0]), int(center[1])-3),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.,
                    color=use_color,
                    thickness=1)
        
        images.append(image)
    return images


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
                # cv2.arrowedLine(img, past_point, p, use_color, thickness=2, tipLength=0.4)
                cv2.line(img, past_point, p, use_color, thickness=2, lineType=cv2.LINE_AA)
            past_point = p

    return img