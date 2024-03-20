#coding: utf-8
#----- 標準ライブラリ -----#
import os

#----- 専用ライブラリ -----#
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import h5py
#----- 自作ライブラリ -----#
#None

def MPM_to_Cell(MPM_data,EP_th=0.3):
    #MPM_data.shape=[frame,3,H,W]
    device = "cuda:1"
    #ガウシアンカーネル定義:kernel_size = 4*sigma+0.5
    gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3).cuda(device)
    #近傍ピクセルより自身が大きいか確かめるためのMaxpooling
    maxpool = nn.MaxPool2d(3, stride=1, padding=1)
    
    #existence probabilityを生成し、細胞位置と移動ベクトルと確信度をとる
    cell_point = []
    batch_size = 128
    for batch in torch.split(MPM_data, batch_size):
        batch = batch.cuda(device, non_blocking=True)
        EP = torch.norm(batch, dim=1)
        EP = gaussian_filter(EP)
        EP = torch.clamp(EP, max=1.0)
        EP_binary = (EP >= EP_th) * (maxpool(EP) == EP) * 1.0
        
        batch = batch.to('cpu').detach().numpy()
        EP = EP.to('cpu').detach().numpy()
        EP_binary = EP_binary.to('cpu').detach().numpy().astype(np.uint8)
        for batch,EP, EP_binary_img in zip(batch, EP, EP_binary):
            _, _, _, center = cv2.connectedComponentsWithStats(EP_binary_img)
            #0番目は背景クラスの重心
            #[cell][y,x]
            center = center[1:].astype(np.int32)
            #[x,y,t][cell]
            vector = batch[:, center[:, 1], center[:, 0]]
            #マイナス成分をなくす
            vector[-1, :] = np.abs(vector[-1, :])
            x = vector[0,:] / (vector[-1,:] + 1e-7)
            y = vector[1,:] / (vector[-1,:] + 1e-7)
            cell_point.append(np.stack([center[:, 0],
                                center[:, 1],
                                center[:, 0] + x,
                                center[:, 1] + y,
                                EP[center[:, 1], center[:, 0]]],
                                axis=1))
    
    #cell_point = [frame, cell, 5] 5つのパラメータはそれぞれ[x,y,pre_x,pre_y,確信度]
    cell_point = [torch.tensor(cell) for cell in cell_point]

    return cell_point


def extremum_search(MPM, nxt_cell, pool_range=5, EP_th=0.3):
    #極大値を探索するためのMaxpooling
    maxpool = nn.MaxPool2d(pool_range, stride=1, padding=(pool_range - 1) // 2)

    #EP_pointの座標の順番をx,yにするためEP.shape=[1,W,H]にする
    EP = torch.norm(MPM, dim=0, keepdim=True).transpose(1, 2)
    EP_binary = (EP >= EP_th) * (maxpool(EP) == EP) * 1.0
    #sparse行列に直し、極値の座標を取得する
    EP_binary = EP_binary.squeeze()
    #EP_point =[x,y][cell] shape=[2,cell_number]
    EP_point = EP_binary.to_sparse()._indices()

    #次のcellの位置vectorと極値の位置との最小距離を計算する。
    EP_point = EP_point.unsqueeze(1)
    nxt_cell_point = nxt_cell[:, 2:4].t().unsqueeze(-1)
    #distance.shape = [nxt_cell nember,EP_point number]
    #最小距離のEP_point indexを取得
    min_dis_index = torch.norm(nxt_cell_point - EP_point, dim=0).min(dim=1)[1]
    
    return EP_point.squeeze()[:, min_dis_index]


def Cell_Tracker(MPM_data, cell_data):
    """Detection結果から細胞をTrackingする関数

    Args:
        MPM_data (torch.tensor)): MPMマップ
        cell_data (list): 細胞位置のリスト。0次元目はframeを表し、1次元目はCellを3次元目は5つのパラメータ[x,y,pre_x,pre_y,確信度]

    Returns:
        torch.tensor[:,5] (cpu): 細胞Tracking結果。1次元目は[frame,id,x,y,parent id]
    """    
    #MPM_data.shape=[frame,3,H,W]
    device = "cuda:1"

    #最初のセルの位置をid付けする。
    #frameの最初は0
    #cell_trackには[frame,id,x,y,parent_id]が保存される。
    cell_track = torch.tensor([[0, idx+1, *cell[2:4].tolist(), -1]
                                for idx, cell in enumerate(cell_data[0])],device=device)
    #セル同士の関連付け
    frame = 0
    for pre_MPM, nxt_cell in zip(MPM_data[:-1], cell_data[1:]):
        frame += 1
        pre_MPM = pre_MPM.cuda(device, non_blocking=True)
        nxt_cell = nxt_cell.cuda(device, non_blocking=True)

        #nxt_cellに関連した過去のcellの位置
        extremum_point = extremum_search(pre_MPM, nxt_cell)
        #cell_trackから1つ前のframeの座標を撮ってくる。
        pre_cell_point = cell_track[cell_track[:, 0] == frame - 1][:, 2:4].t().unsqueeze(-1)
        #min_disは最小距離の距離とcell_track indexを取得
        min_dis = torch.norm(pre_cell_point - extremum_point.unsqueeze(1), dim=0).t().min(dim=1)
        
        """補足
        min_disはcell_trackのidxとextremum_pointをつなげるもので、
        extremum_pointはnxt_cellのidxと繋がっているため、
        結局、cell_trackとnxt_cellをつなげていることになる。難すぎ
        """

        pre_cell_id = cell_track[cell_track[:, 0] == frame - 1][:, 1]
        #関連付けフラグ。どのnxt_cellが関連付けられたのかを示す
        association_flag = min_dis[0] < 10
        #print(extremum_point.shape)
        #print(pre_cell_point.shape)
        #print(min_dis[0].shape)
        #print(nxt_cell[association_flag].shape)

        #関連付け追跡結果
        association_track = torch.tensor([[frame, pre_cell_id[idx], *cell[2:4].tolist(), -1]
                for idx, cell in zip(min_dis[1][association_flag], nxt_cell[association_flag])],
                device=device)

        #発生の追跡結果
        appearance_track = torch.tensor([[frame, cell_track[:,1].max() + idx + 1, *cell[2:4].tolist(), -1]
                for idx, cell in enumerate(nxt_cell[~association_flag])],
                device=device)
        
        if appearance_track.size(0) != 0:
            max_id = max(cell_track[:, 1].max(), appearance_track[:, 1].max())
        else:
            max_id = cell_track[:, 1].max()

        if association_track.size(0) != 0:
            #分裂によるid重複対策
            unique_id, id_count = torch.unique(association_track[:, 1], return_counts=True)
            #print(association_track[:, 1])
            #print(association_track[:, 1].shape)
            #重複していないid
            unique_id = unique_id[id_count == 1].unsqueeze(0)
            change_flag = association_track[:, 1].unsqueeze(-1)
            #change_flag.shape=[association,unique number]
            change_flag = (change_flag == unique_id) * 1.0
            change_idx = torch.arange(max_id + 1, id_count[id_count > 1].sum() + max_id + 1, device=device)
            
            #親id,自身のid更新
            association_track[change_flag.sum(dim=1) == 0, -1] = association_track[change_flag.sum(dim=1) == 0, 1]
            association_track[change_flag.sum(dim=1) == 0, 1] = change_idx
        

        #print(max_id)
        #print(cell_track.shape)
        #print(association_track.shape)
        #print(appearance_track.shape)
        cell_track = torch.cat([cell_track, association_track, appearance_track], dim=0)
        
        #print(cell_track.shape)


        #input()
    return cell_track.to('cpu')


def Cell_Tracker_non_division(MPM_data, cell_data):
    """Detection結果から細胞をTrackingする関数

    Args:
        MPM_data (torch.tensor)): MPMマップ
        cell_data (list): 細胞位置のリスト。0次元目はframeを表し、1次元目はCellを3次元目は5つのパラメータ[x,y,pre_x,pre_y,確信度]

    Returns:
        torch.tensor[:,5] (cpu): 細胞Tracking結果。1次元目は[frame,id,x,y,parent id]
    """    
    #MPM_data.shape=[frame,3,H,W]
    device = "cuda:1"

    #最初のセルの位置をid付けする。
    #frameの最初は0
    #cell_trackには[frame,id,x,y,parent_id]が保存される。
    cell_track = torch.tensor([[0, idx+1, *cell[2:4].tolist(), -1]
                                for idx, cell in enumerate(cell_data[0])],device=device)
    #セル同士の関連付け
    frame = 0
    for pre_MPM, nxt_cell in zip(MPM_data[:-1], cell_data[1:]):
        frame += 1
        pre_MPM = pre_MPM.cuda(device, non_blocking=True)
        nxt_cell = nxt_cell.cuda(device, non_blocking=True)

        #nxt_cellに関連した過去のcellの位置
        extremum_point = extremum_search(pre_MPM, nxt_cell)
        #cell_trackから1つ前のframeの座標を撮ってくる。
        pre_cell_point = cell_track[cell_track[:, 0] == frame - 1][:, 2:4].t().unsqueeze(-1)
        #min_disは最小距離の距離とcell_track indexを取得
        min_dis = torch.norm(pre_cell_point - extremum_point.unsqueeze(1), dim=0).t().min(dim=1)
        
        """補足
        min_disはcell_trackのidxとextremum_pointをつなげるもので、
        extremum_pointはnxt_cellのidxと繋がっているため、
        結局、cell_trackとnxt_cellをつなげていることになる。難すぎ
        """

        pre_cell_id = cell_track[cell_track[:, 0] == frame - 1][:, 1]
        #関連付けフラグ。どのnxt_cellが関連付けられたのかを示す
        association_flag = min_dis[0] < 10
        #print(extremum_point.shape)
        #print(pre_cell_point.shape)
        #print(min_dis[0].shape)
        #print(nxt_cell[association_flag].shape)

        #関連付け追跡結果
        association_track = torch.tensor([[frame, pre_cell_id[idx], *cell[2:4].tolist(), -1]
                for idx, cell in zip(min_dis[1][association_flag], nxt_cell[association_flag])],
                device=device)

        #発生の追跡結果
        appearance_track = torch.tensor([[frame, cell_track[:,1].max() + idx + 1, *cell[2:4].tolist(), -1]
                for idx, cell in enumerate(nxt_cell[~association_flag])],
                device=device)
        
        if appearance_track.size(0) != 0:
            max_id = max(cell_track[:, 1].max(), appearance_track[:, 1].max())
        else:
            max_id = cell_track[:, 1].max()

        #print(max_id)
        #print(cell_track.shape)
        #print(association_track.shape)
        #print(appearance_track.shape)
        cell_track = torch.cat([cell_track, association_track, appearance_track], dim=0)
        
        #print(cell_track.shape)


        #input()
    return cell_track.to('cpu')



def division_process(cell_track):
    print(cell_track.shape)
    for frame in torch.unique(cell_track[:, 0], sorted=True):
        #親idを持つtrack flag
        childlen_track_flag = (cell_track[:, 0] != frame) & (cell_track[:, -1] != -1)
        #親idを取得
        parent_ids = cell_track[childlen_track_flag, -1]
        #親idを持つtrack idを取得
        childlen_ids = cell_track[childlen_track_flag, 1]

        #親idのtrack取得
        parent_track = [cell_track[(cell_track[:, 0] < frame) & (cell_track[:, 1] == parent_id)] for parent_id in parent_ids]
        
        #親trackのidを子trackのidに置き換える
        for idx in range(len(childlen_ids)):
            parent_track[idx][:, 1] = childlen_ids[idx]

        cell_track = torch.cat([cell_track, *parent_track], dim=0)
        print(cell_track.shape)
    
    return cell_track
