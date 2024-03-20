#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import sys
#----- 専用ライブラリ -----#
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.dataset import OISTLoader, OISTLoader_add, collate_delete_PAD
import utils.utils as ut
from utils.Transformer_to_track import Transformer_to_Track, Transformer_to_Track2, Transformer_to_Track3
from utils.evaluation import Object_Tracking
from models.TATR import TATR_3D

############## dataloader関数##############
def dataload():
    ### data augmentation + preprocceing ###
    val_transform = ut.Compose([#ut.RandomCrop(size=(512, 512)),
                                #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])
    
    #OIST
    #val_dataset = OISTLoader(Density="Sim", mode="val", kind="GFP", length=16, transform=val_transform)
    train_dataset = OISTLoader(Staning="GFP", mode="train", split=0, length=16)
    move_limit = get_movelimit(train_dataset)
    val_dataset = OISTLoader_add(root_dir="/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov", Staning="GFP", mode="test", split=0, length=16, transform=val_transform)
    #val_dataset = OISTLoader_add(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning="GFP", mode="test", split=0, length=16, transform=val_transform)

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return val_loader, move_limit


def get_movelimit(dataset, factor=1.1):
    all_tracks = dataset.get_track()

    move_max_list = [[], [], []]
    cash_move_append = [move_list.append for move_list in move_max_list]

    for track in all_tracks:
        # frame sort
        sort_idx = track[:, 0].sort()[1]
        track = track[sort_idx]
        zero_point = track[track[:, 0] == 0]
        aa = torch.unique(zero_point[:, 1], return_counts=True)

        #frame,id,x,y
        unique_id = torch.unique(track[:, 1])
        for target_id in unique_id:
            for i in range(1, 4):
                target_track = track[track[:, 1] == target_id][:, [2, 3]]
                move = target_track[:-i] - target_track[i:]
                move = torch.sqrt((move ** 2).sum(dim=-1))

                if move.shape[0] != 0:
                    cash_move_append[i - 1](move.max())

    del cash_move_append

    move_limit = [factor * max(move) for move in move_max_list]
    return move_limit


############## validation関数 ##############
def val():
    # モデル→推論モード
    model.eval()
    coord_F = None
    #決定論的
    torch.backends.cudnn.deterministic = True
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            # 入力画像をモデルに入力
            vector, coord, coord_F = model.tracking_process(inputs, add_F_dict=coord_F)
            #vector, coord, coord_F = model.tracking_process(inputs, point[:, :, :, [2, 3]], add_F_dict=coord_F)

            Track_transfor.update(vector, coord)

    Track_transfor()
    Track_transfor.save_track()
    Tracking_Eva.update(Track_transfor.get_track(), val_loader.dataset.CP_data)
    acc = Tracking_Eva()

    Track_transfor.reset()

    torch.backends.cudnn.deterministic = False
    with open("acc.txt", mode='a') as f:
        for k in acc.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in acc.values():
            f.write(f"{v}\t")
        f.write("\n")
    for k, v in acc.items():
        print(f"{k}".ljust(20) + f":{v}")
    return acc


val_loader, move_limit = dataload()

device = torch.device('cuda:0')
Track_transfor = Transformer_to_Track(connect_method="N", match_threshold=0.8,move_limit=move_limit)
#Track_transfor = Transformer_to_Track2(connect_method="N",move_limit=move_limit)
#Track_transfor = Transformer_to_Track3(move_limit=move_limit)


Tracking_Eva = Object_Tracking()

model = TATR_3D(back_bone_path="/mnt/kamiya/code/TATR3D/outputs/back_bone/GFP_0/256_100/result/model.pth",
                noise_strength=0.4,
                pos_mode="Sin",
                encode_mode="Time",
                move_limit=move_limit[0]).cuda(device)

model_path = "/mnt/kamiya/code/TATR3D/outputs/GFP_0/Time_Sin/result/model.pth"
model.load_state_dict(torch.load(model_path))

val()
