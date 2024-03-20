#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random as rm
import time
from functools import lru_cache
#----- 専用ライブラリ -----#
import cv2
import h5py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
from pylab import *
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import ListedColormap, BoundaryNorm
#----- 自作ライブラリ -----#


############## main ##############
if __name__ == '__main__':
    path = "/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov/Sample5/video/000.png"
    image = np.array(Image.open(path).convert("L"))[None]
    image = image.astype(np.uint8).transpose(1,2,0)
    image = np.tile(image, [1, 1, 3]) / 255
    
    #pre_track = np.load("track_TATR3D.npy")
    pre_track = np.load("track_MPM.npy")

    pre_track[:, 0] -= pre_track[:, 0].min()
    pre_track = pre_track[pre_track[:, 2] >= 0]

    fig = plt.figure()
    ax = Axes3D(fig)

    color_dict = {"nopredicted": np.array([1., 0., 0.]), "IDs": np.array([0., 1., 0.]), "preditced": np.array([0., 0., 1.])}
    

    # 画像を挿入
    X1, Y1 = ogrid[0:image.shape[0], 0:image.shape[1]]
    ax.plot_surface(X1, Y1, np.atleast_2d(0), rstride=5, cstride=5, facecolors=image, antialiased=True, zorder=0)

    ids = np.unique(pre_track[:, 1])
    for track_id in ids:
        pre = pre_track[pre_track[:, 1] == track_id][:, [0, 2, 3]]
        
        alpha = 1.0
        id_color = np.array([alpha * rm.uniform(0., 1), alpha * rm.uniform(0., 1), alpha * rm.uniform(0., 1)])

        ax.plot(pre[:, 1], pre[:, 2], pre[:, 0], "-", color=id_color, mew=0.5, zorder=9)


    # 各座標軸のラベルの設定
    ax.set_xlim(0,512)
    ax.set_ylim(0,512)
    #ax.set_zlim(0,100)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_zticks([])
    #ax.axis("off")
    # 描画
    plt.savefig("3D-track_MPM.png")