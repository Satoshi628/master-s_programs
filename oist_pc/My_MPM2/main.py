#coding: utf-8
#----- 標準ライブラリ -----#
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import motmetrics as mm

#----- 自作ライブラリ -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.loss import RMSE_Q_Loss
from utils.dataset import C2C12Loader, OISTLoader, PTCLoader
import utils.utils_C2C12 as uc
import utils.utils_OIST as uo
from U_net import UNet
from utils.MPM_to_track import MPM_to_Cell, Cell_Tracker, Cell_Tracker_non_division
from utils.evaluation import Precition_Recall, Object_Tracking


############## dataloader関数##############
def dataload(cfg):
    if cfg.dataset =="C2C12":
        ### data augmentation + preprocceing ###
        train_transform = uc.Compose([uc.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    # ランダムに左右反転(p=probability)
                                    uc.RandomHorizontalFlip(p=0.5),
                                    uc.RandomVerticalFlip(p=0.5)
                                    ])

        val_transform = uc.Compose([uc.RandomCrop(size=(256, 256))])
        
        # dataset = [image, label]
        # datatype = "train" or "val" or "test", transform = augmentation and preprocceing
        train_dataset = C2C12Loader(use_data_path=cfg.C2C12.train_data,
                                    Annotater=cfg.C2C12.Annotater,
                                    use_iter=[1, 3, 5, 7, 9],
                                    transform=train_transform)
        val_dataset = C2C12Loader(use_data_path=cfg.C2C12.eval_data,
                                Annotater=cfg.C2C12.Annotater,
                                use_iter=[9],
                                transform=val_transform)
    #root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov", Staning="GFP", mode="train", split=0,
    if cfg.dataset == "OIST":
        ### data augmentation + preprocceing ###
        train_transform = uo.Compose([uo.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    # ランダムに左右反転(p=probability)
                                    uo.RandomHorizontalFlip(p=0.5),
                                    uo.RandomVerticalFlip(p=0.5)
                                    ])

        val_transform = uo.Compose([])
        
        # dataset = [image, label]
        # datatype = "train" or "val" or "test", transform = augmentation and preprocceing
        train_dataset = OISTLoader(root_dir=cfg.OIST.root_dir,
                                    Staning=cfg.OIST.Staning,
                                    split=cfg.OIST.split,
                                    mode="train",
                                    use_iter=[1, 2, 3],
                                    transform=train_transform)
        val_dataset = OISTLoader(root_dir=cfg.OIST.root_dir,
                                    Staning=cfg.OIST.Staning,
                                    split=cfg.OIST.split,
                                    mode="val",
                                    use_iter=[1],
                                    transform=val_transform)
    
    if cfg.dataset == "PTC":
        ### data augmentation + preprocceing ###
        train_transform = uo.Compose([uo.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    # ランダムに左右反転(p=probability)
                                    uo.RandomHorizontalFlip(p=0.5),
                                    uo.RandomVerticalFlip(p=0.5)
                                    ])

        val_transform = uo.Compose([])
        
        # dataset = [image, label]
        # datatype = "train" or "val" or "test", transform = augmentation and preprocceing
        train_dataset = PTCLoader(root_dir=cfg.PTC.root_dir,
                                    Staning=cfg.PTC.Staning,
                                    Density=cfg.PTC.Density,
                                    split=cfg.PTC.split,
                                    mode="train",
                                    use_iter=[1, 2, 3],
                                    transform=train_transform)
        val_dataset = PTCLoader(root_dir=cfg.PTC.root_dir,
                                    Staning=cfg.PTC.Staning,
                                    Density=cfg.PTC.Density,
                                    split=cfg.PTC.split,
                                    mode="val",
                                    use_iter=[1],
                                    transform=val_transform)
    

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.parameter.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.parameter.val_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader


############## train関数 ##############
def train(model, train_loader, criterion, optimizer, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #GPUモード高速化。入力、labelはfloat32
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 入力画像をモデルに入力
        output = model(inputs)

        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()
    
    return sum_loss/(batch_idx+1)

############## validation関数 ##############
def val(model, val_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0
    
    predict_MPM = torch.tensor([])
    tracking_fnc = Object_Tracking()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)

            # 入力画像をモデルに入力
            output = model(inputs)

            predict_MPM = torch.cat([predict_MPM, output.to("cpu")], dim=0)
            
        #predict_image_save(inputs, output, targets)
        cell = MPM_to_Cell(predict_MPM)
        MPM_track = Cell_Tracker(predict_MPM, cell)
        #MPM_track = Cell_Tracker_non_division(predict_MPM, cell)

        label_cell_track = val_loader.dataset.CP_data
        tracking_fnc.update(MPM_track[:, [0, 1, 2, 3]], label_cell_track)
        tracking_acc = tracking_fnc()
    return tracking_acc

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.parameter.epoch}")
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")

    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/val.txt"

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\t")
        for eval_name in mm.metrics.motchallenge_metrics:
            f.write(f"{eval_name}\t")
        f.write("\n")

    # モデル設定
    model = UNet(n_channels=2, n_classes=3, bilinear=True).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg)

    #criterion = RMSE_Q_Loss(0.8)
    criterion = nn.MSELoss()
    
    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_f1 = 0.
    # args.num_epochs = max epoch
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        if epoch > 70:
            evals = val(model, val_loader, device)
        else:
            evals = {name: 0 for name in mm.metrics.motchallenge_metrics}

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f} val IDF1={:.2%}".format(
                        epoch + 1,
                        cfg.parameter.epoch,
                        train_loss,
                        evals['idf1'])
        
        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch+1}\t")
            for eval_value in evals.values():
                f.write(f"{eval_value}\t")
            f.write("\n")

        if evals['idf1'] > best_f1:
            best_f1 = evals['idf1']
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

    print("最高idf1:{:.2%}".format(best_f1))

############## main ##############
if __name__ == '__main__':
    main()
    
