#coding: utf-8
#----- 標準ライブラリ -----#
import re
import os
import random
import time
from functools import lru_cache
#----- 専用ライブラリ -----#
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


def Visualization_Detection(times=3):
    for batch_idx, (inputs, targets, point) in enumerate(plot_loader):
        if batch_idx >= times:
            break
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        #point = point.cuda(device, non_blocking=True)
        #point = point.long()

        #EP_map出力
        EP_map = model.backbone(inputs[:, :1])
        past_coordinate, present_coordinate, match_map = model(inputs[:, :1], inputs[:, 1:])
        
        torchvision.utils.save_image(inputs[:, :1],
                                    f"result/input/{batch_idx:04}.png",
                                    normalize=True)
        
        torchvision.utils.save_image(EP_map[0],
                                    f"result/predict_EP_image/{batch_idx:04}.png",
                                    normalize=True)
        
        torchvision.utils.save_image(targets[:, :1],
                                    f"result/label_EP_image/{batch_idx:04}.png",
                                    normalize=True)

        coordinate = past_coordinate[0].to('cpu').detach().numpy().copy()
        back_image = EP_map[0][0, 0].to('cpu').detach().numpy().copy()

        back_image = (back_image * 255).astype(np.uint8)

        coordinate = coordinate[coordinate[:, 0] != -1]

        plt.clf()
        plt.imshow(back_image, cmap="gray")
        for coord in coordinate:
            plt.scatter(coord[0], coord[1], s=30, marker="x",c="r")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(f"result/coord_image/{batch_idx:04}.png")

def Visualization_Tracking():
    video = []
    for batch_idx, (inputs, targets, point) in enumerate(plot_loader):
        #channelとlengthを入れ替え、batchとlengthをまとめる
        video.append(inputs.transpose(1, 2).flatten(0, 1))

    #for batch_idx, inputs in enumerate(plot_loader):
    #    #channelとlengthを入れ替え、batchとlengthをまとめる
    #    inputs = inputs[:, :, :-1]
    #    video.append(inputs.transpose(1, 2).flatten(0, 1))

    video = torch.cat(video, dim=0)
    video = video.detach().numpy().copy()
    video = (video * 255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)
    print(video.shape)

    track = np.load("/mnt/kamiya/code/TATR3D/outputs/GFP_0/Time_MLP/track.npy")
    #track = np.load("/mnt/kamiya/code/TATR3D/outputs/VESICLE/1/track.npy").astype(np.int32)
    #track = np.load("/mnt/kamiya/code/TATR3D/outputs/GFP_0_immobile/Time_MLP/track.npy").astype(np.int32)
    #track = np.load("/mnt/kamiya/code/TATR3D/track.npy").astype(np.int32)
    track_ids = np.unique(track[:, 1])

    track = [track[(track[:, 0] == frame) & (track[:, 2] >= 0)]
            for frame in sorted(np.unique(track[:, 0]))]
    
    #labelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    #writer = cv2.VideoWriter('result/video/raw_track.mp4', fmt, 30., (640, 480))
    writer = cv2.VideoWriter('result/video/track.mp4', fmt, 30., (512, 512))

    random_color = {track_id.astype(np.str_): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    for image, pre in zip(video, track):
        image = image.copy()
        for coord in pre:
            use_color = random_color[coord[1].astype(np.str_)]
            image = cv2.circle(image,
                                center=(coord[2], coord[3]),
                                radius=6,
                                color=use_color,
                                thickness=1)
            image = cv2.putText(image,
                                text=coord[1].astype(np.str_),
                                org=(coord[2], coord[3]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=use_color,
                                thickness=1)
        
        writer.write(image)

    #ファイルを閉じる
    writer.release()

def Visualization_detection_and_Tracking():
    TtT = Transformer_to_Track()
    video = []
    detection = []
    # モデル→推論モード
    model.eval()
    #決定論的
    torch.backends.cudnn.deterministic = True
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in enumerate(plot_loader):
            #channelとlengthを入れ替え、batchとlengthをまとめる
            video.append(inputs.transpose(1, 2).flatten(0, 1))

            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            vector, coord = model(inputs)

            #検出位置保存,[num,2(x,y)]の形で保存
            #numにはNone tokenやPADは入っていない
            for p in coord.flatten(0, 1):
                p = p[p[:, 0] >= 0].to('cpu').detach().numpy().copy().astype(np.int16)
                detection.append(p)

            TtT.update(vector, coord)

    TtT()

    #画像処理
    video = torch.cat(video, dim=0)
    video = video.detach().numpy().copy()
    video = (video*255).astype(np.uint8)
    #video[100,512,512]から[100,512,512,3]にする
    video = np.tile(video, (1, 3, 1, 1)).transpose(0, 2, 3, 1)

    #検出処理,すでにリストがframe数、保存されているデータが[num,2(x,y)]のnumpyになっている
    #None


    #[num,4(frame,id,x,y)]
    track = TtT.get_track().detach().numpy().copy().astype(np.int16)

    track_ids = np.unique(track[:, 1])

    track = [track[(track[:, 0] == frame) & (track[:, 2] >= 0)]
                            for frame in np.unique(track[:, 0])]

    #modelのdetectionでのTrackingを動画化
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(
        'result/video/detection_and_track.mp4', fmt, 30., (512, 512))

    random_color = {track_id.astype(np.str_): [random.randint(128, 255) for _ in range(3)] for track_id in track_ids}

    for image, predict, detect in zip(video, track, detection):
        image = image.copy()

        #Tracking可視化
        for coord_track in predict:
            use_color = random_color[coord_track[1].astype(np.str_)]
            image = cv2.circle(image,
                                center=(coord_track[2], coord_track[3]),
                                radius=6,
                                color=use_color,
                                thickness=1)
            image = cv2.putText(image,
                                text=coord_track[1].astype(np.str_),
                                org=(coord_track[2], coord_track[3]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.4,
                                color=use_color,
                                thickness=1)
        
        #Detection可視化
        for coord_detect in detect:
            use_color = [0, 0, 255]
            image = cv2.drawMarker(image,
                                (coord_detect[0], coord_detect[1]),
                                markerType=cv2.MARKER_TILTED_CROSS,
                                markerSize=8,
                                color=use_color,
                                thickness=1)

        writer.write(image)

    #ファイルを閉じる
    writer.release()

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
        Feature_vector1 = F1[:, point[:, 0], point[:, 2], point[:, 1]]
        Feature_vector2 = F2[:, point[:, 0], point[:, 2], point[:, 1]]
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
    
    plot_transform = ut.Compose([])
    #plot_dataset = C2C12Loader(condition="Control",mode="test",Annotater="Computer",transform=test_transform)
    #plot_dataset = OISTLoader(Density="Sim", mode="val", kind="GFP", length=16, transform=plot_transform)
    #plot_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov", Staning="GFP", mode="test", split=0, length=16, transform=plot_transform)
    plot_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning="GFP", mode="test", split=0, length=16, transform=plot_transform)
    #plot_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/Sim_immobile_Density_mov", Staning="GFP_Low", mode="test", split=0, length=16, transform=plot_transform)
    #plot_dataset = PTC_Loader(root_dir="/mnt/kamiya/dataset/PTC", Molecule="VESICLE", Density="Low", mode="test", split=1, length=16, transform=plot_transform)
    plot_loader = torch.utils.data.DataLoader(
        plot_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=4,
        shuffle=False,
        drop_last=False)
    #plot_dataset = Raw_OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/LowDensity_mov/Sample1")
    #
    #plot_loader = torch.utils.data.DataLoader(
    #    plot_dataset,
    #    batch_size=1,
    #    shuffle=False,
    #    num_workers=2,
    #    drop_last=False)
    
    return plot_loader


############## main ##############
if __name__ == '__main__':
    device = torch.device('cuda:0')

    if not os.path.exists("result"):
        os.mkdir("result")

    if not os.path.exists("result/input"):
        os.mkdir("result/input")
    
    if not os.path.exists("result/predict_EP_image"):
        os.mkdir("result/predict_EP_image")
    
    if not os.path.exists("result/label_EP_image"):
        os.mkdir("result/label_EP_image")
    
    if not os.path.exists("result/coord_image"):
        os.mkdir("result/coord_image")
    
    if not os.path.exists("result/video"):
        os.mkdir("result/video")
    
    if not os.path.exists("result/Feature"):
        os.mkdir("result/Feature")
    """
    # モデル設定
    model = TATR_3D_add(noise_strength=0.5,
                    pool_range=11,
                    encode_mode="Learned").cuda(device)

    model_path = "/mnt/kamiya/code/TATR3D2/outputs/OIST_Sim_GFP_3D_FA_len0-1/2021-11-23-15-46/result/model.pth"
    model.load_state_dict(torch.load(model_path))
    """
    # データ読み込み+初期設定
    plot_loader = dataload()

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True
    #決定論的
    torch.backends.cudnn.deterministic = True

    #model.eval()

    #Detectionの可視化
    #Visualization_Detection(times=3)

    #Trackingの可視化
    Visualization_Tracking()

    #TrackingとDetectionの可視化
    #Visualization_detection_and_Tracking()

    #特徴量の可視化
    #Visualization_Featrue()
