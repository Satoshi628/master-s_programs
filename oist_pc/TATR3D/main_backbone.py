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
#----- 自作ライブラリ -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import OISTLoader, collate_delete_PAD
import utils.utils as ut
from utils.evaluation import Object_Detection
from models.Unet_3D import UNet_3D

############## dataloader関数##############
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
    if cfg.dataset == "OIST":
        train_dataset = OISTLoader(Density=cfg.OIST.Density, mode="train", kind=cfg.OIST.kind, length=cfg.OIST.length, transform=train_transform)
        val_dataset = OISTLoader(Density=cfg.OIST.Density, mode="val", kind=cfg.OIST.kind, length=cfg.OIST.length, transform=val_transform)

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        #num_workers=os.cpu_count(),
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.val_size,
        shuffle=False,
        #num_workers=os.cpu_count(),
        drop_last=False)
    
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
    for batch_idx, (inputs, targets, point) in enumerate(tqdm(train_loader, leave=False)):
        #input.size() => [batch,channel,length,H,W]
        #targets.size() => [batch,length,H,W]
        #point.size() => [batch,length,num,4(frame,id,x,y)]

        #GPUモード高速化。入力、labelはfloat32
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 入力画像をモデルに入力
        predicts, feature = model(inputs)

        #二乗誤差ではサイズを合わせないといけないため
        predicts = predicts.squeeze(1)

        loss = criterion(predicts, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()
    
    return sum_loss / (batch_idx + 1)

############## validation関数 ##############
def val(model, val_loader, criterion, val_function, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in enumerate(tqdm(val_loader, leave=False)):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            point = point[:, :, :, 2:].cuda(device, non_blocking=True)
            point = point.long()

            # 入力画像をモデルに入力
            predicts, feature = model(inputs)

            #二乗誤差ではサイズを合わせないといけないため、squeeze(1)
            #detection精度の計算では[batch,1,length,H,W]が必要
            # My_MSEloss or MESloss
            loss = criterion(predicts.squeeze(1), targets)

            #Detectionの精度を計算
            for detection_val in val_function:
                #複数回preditctsが出力されるが、一回だけ精度計算するようにする。
                if batch_idx == 0:
                    #最初の処理ではlengthの半分の出力を計算する
                    for frame_idx in range(predicts.shape[2] // 2):
                        coordinate = detection_val.coordinater(predicts[:, :, frame_idx])  #predictsの大きさは[batch,1,H,W]
                        label_point = point[:, frame_idx]
                        detection_val.calculate(coordinate, label_point)
                else:#中間の処理では8番目(idxでは7)の出力を計算する。
                    coordinate = detection_val.coordinater(predicts[:, :, 7])
                    label_point = point[:, 7]
                    detection_val.calculate(coordinate, label_point)
            # loss溜め
            sum_loss += loss.item()
        
        
        #Detectionの精度を計算
        for detection_val in val_function:
            #最後の処理ではlengthの後半を計算する。
            for frame_idx in range(predicts.shape[2] // 2):
                coordinate = detection_val.coordinater(predicts[:, :, frame_idx + 8])  #predictsの大きさは[batch,1,H,W]
                label_point = point[:, frame_idx + 8]
                detection_val.calculate(coordinate, label_point)

        #Detectionの精度を出力
        detection_accuracy = [detection_val() for detection_val in val_function]
        
    return sum_loss / (batch_idx + 1), detection_accuracy

@hydra.main(config_path="config", config_name='main_backbone.yaml')
def main(cfg):
    print(f"epoch       :{cfg.parameter.epoch}")
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")

    #dataset判定
    if not cfg.dataset in ["C2C12", "OIST"]:
        raise ValueError("hydraのdatasetパラメータが違います.datasetはC2C12,OISTのみ値が許されます.")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')
    
    #評価指標計算クラス定義
    val_function = [Object_Detection(mode="maximul", noise_strength=0.5, pool_range=11),
                    Object_Detection(mode="maximul", noise_strength=0.6, pool_range=11),
                    Object_Detection(mode="maximul", noise_strength=0.7, pool_range=11),
                    Object_Detection(mode="maximul", noise_strength=0.8, pool_range=11)]
    
    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/val.txt"

    with open(PATH_1, mode='w') as f:
        f.write("Epoch\tTrain Loss\n")
    with open(PATH_2, mode='w') as f:
        f.write("Epoch\tVal Loss\n")
    
    PATH_acc = []
    for idx, val_fnc in enumerate(val_function):
        temp_PATH = f"result/accuracy{idx:04}.txt"
        PATH_acc.append(temp_PATH)
        with open(temp_PATH, mode='w') as f:
            for method_key,method_value in vars(val_fnc).items():
                f.write(f"{method_key}\t{method_value}\n")
            f.write("Epoch\tAccuracy\tPrecition\tRecall\tF1_Score\n")

    # モデル設定
    model = UNet_3D(in_channels=1, n_classes=1).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg)

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

    best_acc = 0.
    best_method = 0
    # args.num_epochs = max epoch
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        val_loss, val_accuracy = val(model, val_loader, criterion, val_function, device)

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f} ValLoss={:.5f}".format(
                        epoch + 1,
                        cfg.parameter.epoch,
                        train_loss,
                        val_loss)
        
        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch + 1, val_loss))
        for idx, accuracy in enumerate(val_accuracy):
            with open(PATH_acc[idx], mode='a') as f:
                f.write(f"{epoch+1}\t{accuracy[0]:.2%}\t{accuracy[1]:.2%}\t{accuracy[2]:.2%}\t{accuracy[3]:.2%}\n")
        
        #accuracyの最大値を計算するためにnumpyに変換
        np_acc = np.array(val_accuracy)
        np_acc_max = np.max(np_acc[:, 0])
        np_acc_max_idx = np.argmax(np_acc[:, 0])

        if np_acc_max >= best_acc:
            best_acc = np_acc_max
            best_method = np_acc_max_idx
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

    print(f"最高accuracy:{best_acc:.2%}")
    print("検出方法")
    for key, value in vars(val_function[best_method]).items():
        print(f"{key}:\t\t{value}\n")
    with open(PATH_acc[best_method], mode='a') as f:
        f.write(f"\nBest Method")

############## main ##############
if __name__ == '__main__':
    main()
    
