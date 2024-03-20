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
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.loss import connected_loss
from utils.dataset import OISTLoader, OISTLoader_add, PTC_Loader, collate_delete_PAD
import utils.utils as ut
from utils.Transformer_to_track import Transformer_to_Track, Transformer_to_Track3
from utils.evaluation import Object_Tracking
from models.TATR import TATR_3D



############## dataloader関数 ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                # ランダムに左右反転(p=probability)
                                ut.RandomHorizontalFlip(p=0.5),
                                ut.RandomVerticalFlip(p=0.5),
                                #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    val_transform = ut.Compose([#ut.RandomCrop(size=(512, 512)),
                                #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])
    #OIST
    # Staning="GFP", mode="train", split=0
    if cfg.dataset == "OIST":
        train_dataset = OISTLoader(Staning=cfg.OIST.Staning, mode="train", split=cfg.OIST.split, length=cfg.OIST.length, transform=train_transform)
        val_dataset = OISTLoader_add(root_dir="/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov", Staning=cfg.OIST.Staning, mode="val", split=cfg.OIST.split, length=cfg.OIST.length, transform=val_transform)
        #val_dataset = OISTLoader_add(Staning=cfg.OIST.Staning, mode="val", split=cfg.OIST.split, length=cfg.OIST.length, transform=val_transform)

    if cfg.dataset == "PTC":
        train_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density=cfg.PTC.Density, mode="train", split=cfg.PTC.split, length=cfg.PTC.length, transform=train_transform)
        val_dataset = PTC_Loader(root_dir=cfg.PTC.root_dir, Molecule=cfg.PTC.Molecule, Density="Low", mode="val", split=cfg.PTC.split, length=cfg.PTC.length, add_mode=True, transform=val_transform)


    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.val_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return train_loader, val_loader

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

############## train関数 ##############
def train(model, train_loader, criterion, optimizer, device):
    # モデル→学習モード
    model.train()
    #非決定論的
    torch.backends.cudnn.deterministic = False

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets, point) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #input.size() => [batch,channel,length,H,W]
        #targets.size() => [batch,length,H,W]
        #point.size() => [batch,length,num,4(frame,id,x,y)]

        #GPUモード高速化。入力、labelはfloat32
        inputs = inputs.cuda(device, non_blocking=True)
        #targets = targets.cuda(device, non_blocking=True)
        point = point.cuda(device, non_blocking=True)
        point = point.long()


        # 入力画像をモデルに入力
        #connected_frames, coordinate = model(inputs)
        vector, coordinate = model(inputs, point[:, :, :, [2, 3]])
        
        loss = criterion(vector, point[:, :, :, 1], coordinate)


        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        # 誤差逆伝播を実行後、計算グラフを削除
        del loss
        # パラメーター更新
        optimizer.step()
    
    torch.backends.cudnn.deterministic = True
    return sum_loss / (batch_idx + 1)

############## validation関数 ##############
def val(model, val_loader, criterion, Track_transfor, Tracking_Eva, device):
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
            vector, coord, coord_F = model.tracking_process(inputs, point[:, :, :, [2, 3]], add_F_dict=coord_F)

            Track_transfor.update(vector, coord)
    
    Track_transfor()
    Tracking_Eva.update(Track_transfor.get_track(), val_loader.dataset.CP_data)
    acc = Tracking_Eva()

    Track_transfor.reset()

    torch.backends.cudnn.deterministic = False
    return acc

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.parameter.epoch}")
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")

    #dataset判定
    if not cfg.dataset in ["C2C12", "OIST", "PTC"]:
        raise ValueError("hydraのdatasetパラメータが違います.datasetはC2C12,OIST,PTCのみ値が許されます.")

    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tPrecision\tRecall\tF1 Score\n")
    
    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg)

    move_limit = get_movelimit(train_loader.dataset, factor=1.1)

    # def model
    model = TATR_3D(back_bone_path=cfg.parameter.back_bone,
                            noise_strength=cfg.parameter.noise_strength,
                            assignment=cfg.parameter.assignment,
                            feature_num=cfg.parameter.feature_num,
                            overlap_range=cfg.parameter.overlap_range,
                            pos_mode=cfg.parameter.pos,
                            Encoder=cfg.parameter.encoder,
                            move_limit=move_limit[0]).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    #model_path = "result/model.pth"
    #model.load_state_dict(torch.load(model_path))

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #精度計算クラス定義
    Track_transfor = Transformer_to_Track3(tracking_mode="P", move_limit=move_limit)
    tracking_fnc = Object_Tracking()

    criterion = connected_loss(move_limit)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_F1 = 0.0
    best_loss = 99999.
    # args.num_epochs = max epoch
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        if epoch >= 10:
            Tracking_acc = val(model, val_loader, criterion, Track_transfor, tracking_fnc, device)
        else:
            Tracking_acc = {'mota': 0.0, 'idp': 0.0, 'idr': 0.0, 'idf1': 0.0, 'num_switches': 0}

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f} Precision={:.5f} Recall={:.5f} F1 Score={:.5f} ".format(
            epoch + 1,
            cfg.parameter.epoch,
            train_loss,
            Tracking_acc['idp'], Tracking_acc['idr'], Tracking_acc['idf1'])

        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write("{0}\t{1[mota]:.4f}\t{1[idp]:.4f}\t{1[idr]:.4f}\t{1[idf1]:.4f}\t{1[num_switches]}\n".format(epoch + 1, Tracking_acc))
        
        if Tracking_acc['idf1'] > best_F1:
            best_F1 = Tracking_acc['idf1']
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)
        """
        if train_loss < best_loss:
            best_loss = train_loss
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)
        """

    print("最高F1Score:{:.4f}%".format(best_F1))

############## main ##############
if __name__ == '__main__':
    main()
    
