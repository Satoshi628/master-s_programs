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
import optuna
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

optuna.logging.disable_default_handler()


############## dataloader関数##############
def dataload(split, aug_color, aug_Perspective, aug_Rotation):
    ### data augmentation + preprocceing ###
    augment_list = [ndarrayToTensor(),
                    RandomCrop(size=(800, 800)),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5)]
    
    if aug_color == "Color":
        augment_list.append(ColorJitter(brightness=[0.0,1.0], contrast=[0.0,1.0], saturation=[0.0,1.0], hue=[0.0,0.5]))
    if aug_Perspective == "Perspective":
        augment_list.append(RandomPerspective(distortion_scale=0.5, p=0.5))
    if aug_Rotation == "Rotation":
        augment_list.append(RandomRotation([-45.,45.]))

    augment_list.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

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


############## pre train関数 ##############
def pre_train(split, model, optimizer, margin_pram, device):
    # モデル→学習モード
    model.train()

    train_transform = Compose([ndarrayToTensor(),
                                RandomCrop(size=(800, 800)),  # random crop(945x945 -> 800x800)
                                RandomHorizontalFlip(p=0.5), # left right flip(p=probability)
                                RandomVerticalFlip(p=0.5), # up down flip(p=probability)
                                ColorJitter(brightness=[0.0,1.0], contrast=[0.0,1.0], saturation=[0.0,1.0], hue=[0.0,0.5]),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    
    train_dataset1 = Eye_Loader_ARMS2(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='train', split=split, transform=train_transform)
    train_dataset2 = Eye_Loader_ARMS2(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='train', split=split, transform=train_transform)
    
    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                                batch_size=8,
                                                shuffle=True,
                                                drop_last=True)

    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                                batch_size=8,
                                                shuffle=True,
                                                drop_last=True)

    criterion = Contrastive_Loss(margin=margin_pram)

    # 初期設定
    sum_loss = 0

    for e in range(10):
        # 初期設定
        sum_loss = 0
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(train_loader1, train_loader2)):
            #GPUモード高速化
            inputs1 = inputs1.cuda(device, non_blocking=True)
            targets1 = targets1.cuda(device, non_blocking=True)
            inputs2 = inputs2.cuda(device, non_blocking=True)
            targets2 = targets2.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets1 = targets1.long()

            # 入力画像をモデルに入力
            out1, out2 = model.pre_trainning(inputs1, inputs2)

            # 損失計算
            #output.size() => [batch,1] output[:,0].size() => [batch]
            loss = criterion(out1, out2, targets1, targets2)

            # 勾配を0に
            optimizer.zero_grad()

            # 誤差逆伝搬
            loss.backward()

            # loss溜め
            sum_loss += loss.item()

            del loss  # 誤差逆伝播を実行後、計算グラフを削除

            # パラメーター更新
            optimizer.step()
        
        sum_loss = sum_loss/(batch_idx+1)
        print(f"{e+1}/{10}epoch: train Loss:{sum_loss:.3f}")


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


def split_acc(split,aug_color, aug_Perspective, aug_Rotation, model_name, train_name, Lossese, **kwargs):
    device = torch.device('cuda:0')

    train_loader, val_loader, test_loader = dataload(split, aug_color, aug_Perspective, aug_Rotation)

    if model_name == "Resnet18":
        if train_name == "None":
            model = resnet2D18()
        if train_name == "Pretrained":
            model = resnet18_pretrained()
        if train_name == "Siam":
            model = resnet2D18_siam()
    
    if model_name == "Resnet50":
        if train_name == "None":
            model = resnet2D50()
        if train_name == "Pretrained":
            model = resnet50_pretrained()
        if train_name == "Siam":
            model = resnet2D50_siam()

    if Lossese == "CE":
        criterion = nn.CrossEntropyLoss()

    if Lossese == "Focal":
        criterion = FocalLoss(gamma=kwargs["gamma_pram"])
    
    model = model.cuda(device)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam

    if train_name == "Siam":
        pre_train(split, model, optimizer, kwargs["margin_pram"], device)

    print(model_name, train_name, Lossese, aug_color, aug_Perspective, aug_Rotation, kwargs)


    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=20,
                                              cycle_mult=2,
                                              max_lr=0.001,
                                              min_lr=0.0,
                                              warmup_steps=0,
                                              gamma=0.5)

    best_acc = 0.0
    best_test_acc = 0.0
    # args.num_epochs = max epoch
    for epoch in range(40):
        # train
        train(model, train_loader, criterion, optimizer, device)
        # validation
        acc = val(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc
            print(f"{epoch+1}/{60}epoch: best acc:{acc:.2%}")
            best_test_acc = test(model, test_loader, device)

        scheduler.step()

    return best_acc

def objective(trial):
    # 変更パラメータ定義
    model_name = trial.suggest_categorical("model", ["Resnet18", "Resnet50"])
    #model_name = trial.suggest_categorical("model", ["Resnet50"])
    train_name = trial.suggest_categorical("train", ["None", "Pretrained", "Siam"])
    #train_name = trial.suggest_categorical("train", ["Pretrained"])

    Lossese = trial.suggest_categorical("Loss", ["CE", "Focal"])

    aug_color = trial.suggest_categorical("ColorJitter", ["None", "Color"])
    aug_Perspective = trial.suggest_categorical("Perspective", ["None", "Perspective"])
    aug_Rotation = trial.suggest_categorical("Rotation", ["None", "Rotation"])

    param_kwargs ={}

    if Lossese == "Focal":
        gamma_pram = trial.suggest_uniform('gamma', 0., 2.)
        param_kwargs["gamma_pram"]=gamma_pram
    if train_name == "Siam":
        margin_pram = trial.suggest_uniform('margin', 1., 5.)
        param_kwargs["margin_pram"]=margin_pram

    all_acc = 0
    for split in range(5):
        print(split)
        acc = split_acc(split, aug_color, aug_Perspective, aug_Rotation, model_name, train_name, Lossese, **param_kwargs)
        all_acc += acc
    
    best_acc = all_acc/5

    return -best_acc


############## main ##############
if __name__ == '__main__':
    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # pythone
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # 高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    TRIAL_SIZE = 100
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    with open(f"optuna_{split}.txt", mode='w') as f:
        for key, value in study.best_params.items():
            f.write(f"{key}\t{value}\n")

        f.write(f"best IoU:{-study.best_value}")