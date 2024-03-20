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
from utils.dataset import OISTLoader, PTC_Loader, collate_delete_PAD, OISTLoader_add
import utils.utils as ut
from utils.Transformer_to_track import Transformer_to_Track, Transformer_to_Track2, Transformer_to_Track3
from utils.evaluation import Object_Detection,Object_Tracking
from models.TATR import TATR_3D

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    
    test_transform = ut.Compose([
                                #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])
    
    #OIST
    if cfg.dataset == "OIST":
        test_dataset = OISTLoader_add(root_dir=cfg.OIST.root_dir, Staning=cfg.OIST.Staning, mode="test", split=cfg.OIST.split, length=cfg.OIST.length, transform=test_transform)
        #test_dataset = OISTLoader_add(root_dir="/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov", Staning=cfg.OIST.Staning, mode="test", split=cfg.OIST.split, length=cfg.OIST.length, transform=test_transform)
        mode_limit_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning=cfg.OIST.Staning, mode="train", split=cfg.OIST.split, length=cfg.OIST.length)
        move_limit = get_movelimit(mode_limit_dataset)
    if cfg.dataset == "PTC":
        test_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density=cfg.PTC.Density, mode="test", split=cfg.PTC.split, length=cfg.PTC.length, add_mode=True, transform=test_transform)
        mode_limit_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density="Mid", mode="train", split=cfg.PTC.split, length=cfg.PTC.length)
        move_limit = get_movelimit(mode_limit_dataset)
    
    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_limit


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
def test(model, test_loader,  device):
    # モデル→推論モード
    model.eval()
    coord_F = None
    detect = []

    start = time.time()
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            # 入力画像をモデルに入力
            coord = model.backbone.get_coord(inputs)
            #vector, coord, coord_F = model.tracking_process(inputs, point[:, :, :, [2, 3]], add_F_dict=coord_F)
            detect.extend(coord)
    
    detect = [np.concatenate([np.full([c.shape[0],1], idx), c],axis=-1) for idx, c in enumerate(detect)]
    detect = np.concatenate(detect, axis=0)
    bbsize = 11
    print(detect)
    detect = np.concatenate([detect[:, :1], detect[:, 1:3] - bbsize / 2, detect[:, 1:3] + bbsize / 2, detect[:, -1:]], axis=-1)
    print(detect)
    fmt = ["%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.2f"]
    np.savetxt("GFP-Mid.txt", detect, fmt=fmt, delimiter=",")

    return 

@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")

    #dataset判定
    if not cfg.dataset in ["C2C12", "OIST", "PTC"]:
        raise ValueError("hydraのdatasetパラメータが違います.datasetはC2C12,OIST,PTCのみ値が許されます.")

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
                    Encoder=cfg.parameter.encoder,
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

    test(model, test_loader, device)


############## main ##############
if __name__ == '__main__':
    main()
