#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import time
from functools import lru_cache
#----- 専用ライブラリ -----#
from tqdm import tqdm
import cv2
from pylab import *
import h5py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import random
import numbers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE

#----- 自作ライブラリ -----#
from utils.dataset import OISTLoader, PTC_Loader, collate_delete_PAD, Raw_OISTLoader
import utils.utils as ut
from models.TATR import TATR_3D
from utils.Transformer_to_track import Transformer_to_Track

def Feature_hist():
    model.eval()

    hist_data = torch.empty([0])

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(plot_loader), total=len(plot_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            # 入力画像をモデルに入力
            vector = model.feature_hist(inputs, point[:, :, :, [2, 3]])
            vector = vector.flatten().cpu().detach()

            hist_data = torch.cat([hist_data, vector], dim=0)
    
    print(hist_data.shape)

    plt.hist(hist_data, bins=100, range=[0,40])
    plt.savefig("RFAMなし.png")


def Visualization_Featrue():
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in enumerate(plot_loader):
            #GPUモード高速化
            inputs = inputs[:1].cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)
            point = point[:1].cuda(device, non_blocking=True)
            point = point.long()

            F1, F2 = model.backbone.compare_feature(inputs, point[:, :, :, [2, 3]])
            break
        point = point.squeeze()
        F1 = F1.squeeze()
        F2 = F2.squeeze()

        point = point[point[:, :, 0] >= 0]

        Feature_vector1 = F1[:, point[:, 0], point[:, 3], point[:, 2]]
        Feature_vector2 = F2[:, point[:, 0], point[:, 3], point[:, 2]]
        Feature_vector1 = Feature_vector1.transpose(0, 1)
        Feature_vector2 = Feature_vector2.transpose(0, 1)


        emb1 = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(Feature_vector1, device="cuda:0")
        emb2 = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(Feature_vector2, device="cuda:0")

        point = point.to('cpu').detach().numpy().copy()

        track_ids = np.unique(point[:, 1])

        random_color = {track_id.astype(np.str_): [random.random() for _ in range(3)] for track_id in track_ids}


        plt.clf()
        plt.title("3DUnet feature before add feature")
        for track_id, coord in zip(point[:, 1], emb1):
            use_color = random_color[track_id.astype(np.str_)]
            plt.plot(coord[0], coord[1], marker=".", color=use_color)

        plt.savefig(f"result/Feature/before_add.png")

        plt.clf()
        plt.title("3DUnet feature after add feature")
        for track_id, coord in zip(point[:, 1], emb2):
            use_color = random_color[track_id.astype(np.str_)]
            plt.plot(coord[0], coord[1], marker=".", color=use_color)

        plt.savefig(f"result/Feature/after_add.png")


def Visualization_Featrue():
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in enumerate(plot_loader):
            #GPUモード高速化
            inputs = inputs[:1].cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)
            point = point[:1].cuda(device, non_blocking=True)
            point = point.long()

            F1, F2 = model.backbone.compare_feature(inputs, point[:, :, :, [2, 3]])
            break
        point = point.squeeze()
        F1 = F1.squeeze()
        F2 = F2.squeeze()

        point = point[point[:, :, 0] >= 0]

        Feature_vector1 = F1[:, point[:, 0], point[:, 3], point[:, 2]]
        Feature_vector2 = F2[:, point[:, 0], point[:, 3], point[:, 2]]
        Feature_vector1 = Feature_vector1.transpose(0, 1)
        Feature_vector2 = Feature_vector2.transpose(0, 1)


        emb1 = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(Feature_vector1, device="cuda:0")
        emb2 = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(Feature_vector2, device="cuda:0")

        point = point.to('cpu').detach().numpy().copy()

        track_ids = np.unique(point[:, 1])

        random_color = {track_id.astype(np.str_): [random.random() for _ in range(3)] for track_id in track_ids}


        plt.clf()
        plt.title("3DUnet feature before add feature")
        for track_id, coord in zip(point[:, 1], emb1):
            use_color = random_color[track_id.astype(np.str_)]
            plt.plot(coord[0], coord[1], marker=".", color=use_color)

        plt.savefig(f"result/Feature/before_add.png")

        plt.clf()
        plt.title("3DUnet feature after add feature")
        for track_id, coord in zip(point[:, 1], emb2):
            use_color = random_color[track_id.astype(np.str_)]
            plt.plot(coord[0], coord[1], marker=".", color=use_color)

        plt.savefig(f"result/Feature/after_add.png")

############## dataloader関数##############
def dataload():
    ### data augmentation + preprocceing ###
    
    plot_transform = ut.Compose([ut.RandomCrop(size=(128, 128))])
    plot_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning="GFP", mode="test", split=0, length=16, transform=plot_transform)
    
    plot_loader = torch.utils.data.DataLoader(
        plot_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        drop_last=False)
    
    return plot_loader


############## main ##############
if __name__ == '__main__':
    device = torch.device('cuda:0')

    if not os.path.exists("result"):
        os.mkdir("result")

    # def model
    model = TATR_3D(back_bone_path="/mnt/kamiya/code/TATR3D/outputs/back_bone/GFP_0/no_assign/result/model.pth",
                    noise_strength=0.4,
                    assignment=False,
                    pos_mode="MLP",
                    feature_num=256,
                    overlap_range=100,
                    encode_mode="Time").cuda(device)

    #.が入るとだめっぽい
    model_path = "/mnt/kamiya/code/TATR3D/outputs/GFP_0/no_assign/result/model.pth"
    """
    # def model
    model = TATR_3D(back_bone_path="/mnt/kamiya/code/TATR3D/outputs/back_bone/GFP_0/256_50/result/model.pth",
                    noise_strength=0.4,
                    assignment=True,
                    pos_mode="MLP",
                    feature_num=256,
                    overlap_range=100,
                    encode_mode="Time").cuda(device)

    #.が入るとだめっぽい
    model_path = "/mnt/kamiya/code/TATR3D/outputs/GFP_0/Time_MLP/result/model.pth"
    """
    model.load_state_dict(torch.load(model_path))

    # データ読み込み+初期設定
    plot_loader = dataload()

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True
    #決定論的
    torch.backends.cudnn.deterministic = True

    #Trackingの可視化
    #Feature_hist()
    Visualization_Featrue()