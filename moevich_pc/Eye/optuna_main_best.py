#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import (Compose,
    Normalize,
    Resize,
    RandomCrop,
    CenterCrop,
    ColorJitter,
    Pad,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomPerspective,
    RandomRotation)


#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_ARMS2
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss


############## dataloader関数##############
def dataload(split):
    ### data augmentation + preprocceing ###
    augment_list = [ndarrayToTensor(),
                    RandomCrop(size=(800, 800)),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    RandomPerspective(distortion_scale=0.5, p=0.5),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    train_transform = Compose(augment_list)

    val_transform = Compose([ndarrayToTensor(),
                                CenterCrop(size=(800,800)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_dataset = Eye_Loader_ARMS2(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='train', split=split, transform=train_transform)
    val_dataset = Eye_Loader_ARMS2(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='val', split=split, transform=val_transform)
    test_dataset = Eye_Loader_ARMS2(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='test', split=split, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=8,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=2)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    # これらをやると
    # num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader, test_loader


############## train関数 ##############
def train(model, train_loader, criterion, optimizer, device):
    # モデル→学習モード
    model.train()

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()

    return


############## val関数 ##############
def val(model, val_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    total = 0
    correct = 0
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()


            # 入力画像をモデルに入力
            output = model(inputs)

            ###精度の計算###
            total += output.size(0)
            correct += (targets == output.max(dim=-1)[1]).sum()
    
    return (correct/total).item()


############## val関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    total = 0
    correct = 0
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output = model(inputs)

            ###精度の計算###
            total += output.size(0)
            correct += (targets == output.max(dim=-1)[1]).sum()
    
    return (correct/total).item()

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    train_loader, val_loader, test_loader = dataload(cfg.dataset.split)

    device = torch.device('cuda:0')


    model = resnet2D50()

    gamma_pram = 1.9691038829992942
    criterion = FocalLoss(gamma=gamma_pram)
    
    model = model.cuda(device)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=30,
                                              cycle_mult=2,
                                              max_lr=0.001,
                                              min_lr=0.0,
                                              warmup_steps=0,
                                              gamma=0.5)

    best_acc = 0.0
    best_test_acc = 0.0
    # args.num_epochs = max epoch
    for epoch in range(60):
        # train
        train(model, train_loader, criterion, optimizer, device)
        # validation
        acc = val(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc
            print(f"{epoch+1}/{300}epoch: best acc:{acc:.2%}")
            best_test_acc = test(model, test_loader, device)

        scheduler.step()
    print(best_test_acc)
    with open(f"optuna_best.txt", mode='a') as f:
            f.write(f"{best_test_acc}\n")


############## main ##############
if __name__ == '__main__':
    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # 高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    main()
