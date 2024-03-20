import os
from time import sleep

from glob import glob
import cv2
from moviepy.editor import *
import numpy as np


def crop_img(imgpath, img_size):
    #H,W,3
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    top, bottom, left, right = np.random.randint(0, 10, [4]) 
    
    img = img[left:-right, top:-bottom]
    W_ratio = img_size[1] / img.shape[1]

    img = cv2.resize(img, None, None, fx=W_ratio, fy=W_ratio)

    pad_H = max(img_size[0] - img.shape[0], 0)
    pad_W = max(img_size[1] - img.shape[1], 0)

    T = pad_H//2 if pad_H%2==0 else pad_H//2+1
    B = pad_H//2

    img = cv2.copyMakeBorder(img, T, B, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
    return img

def make_clip(folder_path, img_size):
    img_paths = glob(os.path.join(folder_path, "*"))
    img_size = max_img_size(img_paths)
    clips = []
    for path in img_paths:
        img = crop_img(path, img_size)
        print(img.shape)
        clips.append(ImageClip(img).set_duration('00:00:01'))

    return clips

def max_img_size(folder_path):
    H_max, W_max = 0, 0
    for path in folder_path:
        H, W = cv2.imread(path).shape[:2]
        if H_max < H:
            H_max = H
        if W_max < W:
            W_max = W
    
    return [H_max, W_max]

def make_clips_video(clips):
    times = 72 // len(clips)
    clips = clips * times
    video = concatenate_videoclips(clips)
    video.write_videofile(
        "test_clips_video.mp4",
        codec='libx264', 
        # audio_codec='aac', 
        # temp_audiofile='temp-audio.m4a', 
        remove_temp=True,
        fps=30
    )


img_size = [1080, 1920]

sleep(3)
clips = make_clip("C:\\Users\\hotta_mini\\affi_data\\tiktok_data\\oreco093", img_size)
make_clips_video(clips)