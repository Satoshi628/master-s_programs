import os
import sys
import argparse

from moviepy.editor import *
import pandas as pd

LIMIT_video_time = 60*2

crop_start_time = 7.

def crop_video(file_path, output_path):
    video = VideoFileClip(file_path)

    #冒頭をカット
    video = video.subclip(crop_start_time, video.end)

    video_time = video.end - video.start
    #カットしても2分以上なら最後から2分を取得
    if video_time > 2*60.:
        video = video.subclip(video.end-2*60., video.end)

    video.write_videofile(
        output_path,
        codec='libx264', 
        audio_codec='aac', 
        temp_audiofile='temp-audio.m4a', 
        remove_temp=True
    )

def delete_video(file_path):
    if  os.path.isfile(file_path):
        os.remove(file_path)
