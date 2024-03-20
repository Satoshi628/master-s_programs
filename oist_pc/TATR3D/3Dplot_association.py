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


def F1_calculate(predict_cell_track, label_cell_track):
    """Precition,Recall,F1 Scoreを計算する関数。trackはframeでソートされてなければならない

    Args:
        predict_cell_track (torch.tensor[:,4]): モデルの追跡結果。1次元目は[frame,id,x,y]
        label_cell_track (torch.tensor[:,4]): 正解ラベルの追跡。1次元目は[frame,id,x,y]

    Returns:
        Precition (float): モデル出力のPrecition精度
        Recall (float): モデル出力のRecall精度
        F1 (float): モデル出力のF1精度
    """        
    #pre_len = predict_cell_track[:, 0].max() - predict_cell_track[:, 0].min() + 1
    #lab_len = label_cell_track[:, 0].max() - label_cell_track[:, 0].min() + 1
    #assert pre_len == lab_len, "predictとlabelのフレーム数が違います:predict={} label={}".format(pre_len, lab_len)
    predict_cell_track[:, 0] -= predict_cell_track[:, 0].min()
    label_cell_track[:, 0] -= label_cell_track[:, 0].min()



    FP = FP_FN_calculate(predicts=predict_cell_track, label=label_cell_track, mode="FP")
    FN = FP_FN_calculate(predicts=predict_cell_track, label=label_cell_track, mode="FN")
    
    #FNとTPの次元の意味を同じにする
    FN = FN.t()

    # 最小の物を抽出しIDFP,IDFNにする
    #matrix.size() => [predict ids, label ids]
    matrix = (FP + FN).to('cpu').detach().numpy().copy()
    row, col = linear_sum_assignment(matrix)
    
    predicts_ids = torch.unique(predict_cell_track[:, 1]).long().tolist()
    label_ids = torch.unique(label_cell_track[:, 1]).long().tolist()
    return predicts_ids, label_ids, row, col

def FP_FN_calculate(predicts, label, mode="FP"):
        """FPとFNを計算するための関数。modeで切り替え可能

        Args:
            predicts (torch.tensor): 細胞の追跡結果
            label (torch.tensor): 細胞追跡の正解ラベル
            mode (str, optional): モードを切り替えれる。FNのときは"FN"と記載. Defaults to "FP".

        Returns:
            torch.tensor[predict_cell_id,label_cell_id]: 計算結果、FP,FNの数を示す
        """        
        if mode == "FN":
            temp = predicts
            predicts = label
            label = temp
        predicts_ids = torch.unique(predicts[:, 1])
        label_ids = torch.unique(label[:, 1])

        FP_FN = torch.zeros((predicts_ids.size(0), label_ids.size(0)))

        for pre_idx, predicts_id in enumerate(predicts_ids):
            for lab_idx, label_id in enumerate(label_ids):
                predict_cell = predicts[predicts[:, 1] == predicts_id]
                label_cell = label[label[:, 1] == label_id]
                # TP_count.size()=>[predict_cell frame,label_cell frame]
                TP_count = predict_cell[:, 0].view(-1, 1) == label_cell[:, 0].view(1, -1)

                count = predict_cell.size(0) - TP_count.sum()

                distance = torch.norm(predict_cell[TP_count.any(1)][:, [2, 3]]
                                    - label_cell[TP_count.any(0)][:, [2, 3]], dim=-1)

                count += (distance > 10.).sum()
                FP_FN[pre_idx, lab_idx] = count

        return FP_FN

def distance(track, p):
    dist = (track[:, 1:] - p[1:]) ** 2
    dist = np.sqrt(np.sum(dist, axis=-1))
    return dist

############## main ##############
if __name__ == '__main__':
    path = "/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov/Sample5/video/000.png"
    image = np.array(Image.open(path).convert("L"))[None]
    image = image.astype(np.uint8).transpose(1,2,0)
    image = np.tile(image, [1, 1, 3]) / 255
    
    #pre_track = np.load("track_TATR.npy")
    pre_track = np.load("track_MPM.npy")

    label_track = h5py.File("/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov/Cell_Point_Annotation.hdf5", mode='r')["Sample5"][...]
    
    pre_track[:, 0] -= pre_track[:, 0].min()
    label_track[:, 0] -= label_track[:, 0].min()

    pre_track = pre_track[pre_track[:, 2] >= 0]
    label_track = label_track[label_track[:, 2] >= 0]
    
    pre = torch.from_numpy(pre_track)
    label = torch.from_numpy(label_track)
    
    predicts_ids, label_ids, row, col = F1_calculate(pre,label)

    fig = plt.figure()
    ax = Axes3D(fig)

    color_dict = {"False": np.array([1.,0.,0.]), "IDs": np.array([0.,0.,1.]), "True": np.array([0.,1.,0.])}

    # 画像を挿入
    X1, Y1 = ogrid[0:image.shape[0], 0:image.shape[1]]
    ax.plot_surface(X1, Y1, np.atleast_2d(0), rstride=5, cstride=5, facecolors=image, antialiased=True, zorder=0)

    #マッチしたtrack
    for pre_idx, label_idx in zip(row, col):
        pre_id = predicts_ids[pre_idx]
        label_id = label_ids[label_idx]
        
        pre = pre_track[pre_track[:, 1] == pre_id][:, [0, 2, 3]]
        label = label_track[label_track[:, 1] == label_id][:, [0, 2, 3]]
        TP = np.array([np.any((label[:, 0] == point[0]) & (distance(label, point) < 10.)) for point in pre])

        alpha = 0.4
        id_color = np.array([alpha * rm.uniform(-1., 1), alpha * rm.uniform(-1., 1), alpha * rm.uniform(-1., 1)])

        for i in range(pre.shape[0]):
            color = color_dict["True"] if TP[i] else color_dict["False"]
            color = np.clip(id_color + color, 0., 1.0)

            ax.plot(pre[i: i + 2, 1], pre[i: i + 2, 2], pre[i: i + 2, 0], "-", color=color, mew=0.5, zorder=9)


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