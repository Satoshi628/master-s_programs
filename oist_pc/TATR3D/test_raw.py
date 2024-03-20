#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.dataset import OISTLoader, PTC_Loader, OISTLoader_add, Raw_OISTLoader
import utils.utils as ut
from utils.Transformer_to_track import Transformer_to_Track, Transformer_to_Track2, Transformer_to_Track3
from utils.evaluation import Object_Detection,Object_Tracking
from models.TATR import TATR_3D

############## dataloader関数##############
def dataload(cfg):
    test_dataset = Raw_OISTLoader(root_dir=cfg.OIST.root_dir)
    mode_limit_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning="GFP", mode="train", split=0, length=16)
    move_limit = get_movelimit(mode_limit_dataset)
    
    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_limit


def get_movelimit(dataset, factor=1.0):
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
def test(model, test_loader, TtT, device):
    # モデル→推論モード
    model.eval()
    coord_F = None

    #精度計算クラス定義
    tracking_fnc = Object_Tracking()

    start = time.time()
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True).float()

            # 入力画像をモデルに入力
            vector, coord, coord_F = model.tracking_process(inputs, add_F_dict=coord_F)
            #Trackingの精度を更新
            TtT.update(vector, coord)
        
        end = time.time()
        print("time", end - start)

        #tracking精度計算 model検出
        TtT()
        TtT.save_track(path="raw_track")


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    # データ読み込み+初期設定
    test_loader, move_limit = dataload(cfg)

    # def model
    model = TATR_3D(back_bone_path=cfg.parameter.back_bone,
                    noise_strength=cfg.parameter.noise_strength,
                    pos_mode=cfg.parameter.pos,
                    assignment=cfg.parameter.assignment,
                    feature_num=cfg.parameter.feature_num,
                    overlap_range=cfg.parameter.overlap_range,
                    encode_mode=cfg.parameter.encoder,
                    move_limit=move_limit[0]).cuda(device)

    #.が入るとだめっぽい
    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    move_limit = [move_limit[0]/(i+1) for i in range(3)]
    print(move_limit)

    #TtT = Transformer_to_Track(move_limit=move_limit, connect_method="N")
    #TtT = Transformer_to_Track2(connect_method="N", move_limit=move_limit)
    TtT = Transformer_to_Track3(tracking_mode="P", move_limit=move_limit)

    test(model, test_loader, TtT, device)



############## main ##############
if __name__ == '__main__':
    main()
