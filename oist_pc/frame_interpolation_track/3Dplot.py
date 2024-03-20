#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random as rm
#----- 専用ライブラリ -----#
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pylab import *

#----- 自作ライブラリ -----#


def get_image(path):
    
    # 動画ファイル読込
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        # 正常に読み込めたのかチェックする
        # 読み込めたらTrue、失敗ならFalse
        print("動画の読み込み失敗")
        sys.exit()
    
    # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
    is_image, frame_img = cap.read()

    cap.release()
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2GRAY)
    
    frame_img = frame_img.astype(np.uint8)[:,:,None]
    frame_img = np.tile(frame_img, [1, 1, 3]) / 255
    return frame_img

############## main ##############
if __name__ == '__main__':
    path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/outputs/GFP_new/0/result/video.mp4"
    image = get_image(path)
    
    path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/outputs/GFP_new/0/result/track.txt"
    path = "/mnt/kamiya/dataset/OIST2/GFP_v9_Mid/10/det.txt"
    pre_path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/track_PTGT.npy"
    pre_path = "/mnt/kamiya/code/my_exp/frame_interpolation_track/track_MPM.npy"
    pre_track = np.load(pre_path)
    print(pre_track)
    pre_track = pre_track[pre_track[:,2] != -2]
    #pre_track = np.load("track_TATR3D.npy")
    # pre_track = np.load("track_TATR3D_PTC.npy")
    # pre_track = np.loadtxt(path, delimiter=",")
    #pre_track = np.load("track_MPM.npy")

    pre_track[:, 0] -= pre_track[:, 0].min()
    # pre_track = pre_track[pre_track[:, 2] >= 0]
    # pre_track = pre_track[pre_track[:, 0] <= 90]

    fig = plt.figure()
    ax = Axes3D(fig)

    # 画像を挿入
    X1, Y1 = ogrid[0:image.shape[0], 0:image.shape[1]]
    ax.plot_surface(X1, Y1, np.atleast_2d(0), rstride=5, cstride=5, facecolors=image, antialiased=True, zorder=0)

    ids = np.unique(pre_track[:, 1])
    # ids = ids[::10]
    for track_id in ids:
        pre = pre_track[pre_track[:, 1] == track_id][:, [0, 2, 3]]
        
        alpha = 1.0
        id_color = np.array([alpha * rm.uniform(0., 1), alpha * rm.uniform(0., 1), alpha * rm.uniform(0., 1)])
        ax.plot(pre[:, 1], pre[:, 2], pre[:, 0], "-", color=id_color, mew=0.5, zorder=9)


    # 各座標軸のラベルの設定
    # ax.set_xlim(0,512)
    # ax.set_ylim(0,512)
    # ax.set_zlim(0,90)
    # ax.set_xticks([])
    # ax.set_yticks([])
    #ax.set_zticks([])
    #ax.axis("off")
    # 描画
    plt.savefig("3D-track_PTC.png")