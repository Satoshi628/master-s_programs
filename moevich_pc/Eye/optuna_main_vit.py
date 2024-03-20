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
import timm

#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_OCT_VIT, Eye_Loader_OCT_VIT_EDIT
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss
from utils.evaluation import Precition_Recall

optuna.logging.disable_default_handler()


############## dataloader関数##############
def dataload(split, aug_color, aug_Perspective, aug_Rotation):
    ### data augmentation + preprocceing ###

    augment_list = [ndarrayToTensor(),
                    Resize([315,315]),
                    RandomCrop(size=(288, 288)),
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
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_dataset = Eye_Loader_OCT_VIT_EDIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='train', split=split, transform=train_transform)
    val_dataset = Eye_Loader_OCT_VIT_EDIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='val', split=split, transform=val_transform)
    test_dataset = Eye_Loader_OCT_VIT_EDIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='test', split=split, transform=val_transform)
    
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
    for batch_idx, (inputs, OCT, targets) in enumerate(train_loader):
        # GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        OCT = OCT.float().cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # H方向に画像を結合
        inputs = torch.cat([inputs, OCT], dim=-2)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output, attens = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()


############## val関数 ##############
def val(model, val_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    acc_calc = Precition_Recall()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, OCT, targets) in enumerate(val_loader):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            inputs = torch.cat([inputs, OCT], dim=-2)


            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs)

            ###精度の計算###
            acc_calc.update(output, targets)
    
    Accuracy, Precition, Recall = acc_calc()
    F1_score = 2*Precition*Recall/(Precition + Recall + 1e-7)
    return F1_score.mean().item()


############## val関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    acc_calc = Precition_Recall()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, OCT, targets) in enumerate(test_loader):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            inputs = torch.cat([inputs, OCT], dim=-2)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs)

            ###精度の計算###
            acc_calc.update(output, targets)
    
    Accuracy, Precition, Recall = acc_calc()
    F1_score = 2*Precition*Recall/(Precition + Recall + 1e-7)
    return F1_score.mean().item()


def split_acc(split,aug_color, aug_Perspective, aug_Rotation, model_name, pretrained, Lossese, **kwargs):
    device = torch.device('cuda:0')

    train_loader, val_loader, test_loader = dataload(split, aug_color, aug_Perspective, aug_Rotation)

    if Lossese == "CE":
        criterion = nn.CrossEntropyLoss()

    if Lossese == "Focal":
        criterion = FocalLoss(gamma=kwargs["gamma_pram"])
    
    model = timm.create_model(model_name, img_size=[576,288], pretrained=pretrained, num_classes=4)
    model = model.cuda(device)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam

    print(model_name, pretrained, Lossese, aug_color, aug_Perspective, aug_Rotation, kwargs)


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
    for epoch in range(10):
        # train
        train(model, train_loader, criterion, optimizer, device)
        # validation
        acc = val(model, val_loader, device)

        if acc > best_acc:
            best_acc = acc
            print(f"{epoch+1}/{60}epoch: best acc:{acc:.2%}")
            best_test_acc = test(model, test_loader, device)

        scheduler.step()

    return best_test_acc

def objective(trial):
    # 変更パラメータ定義
    model_name = trial.suggest_categorical("model", ["vit_tiny_patch16_224_in21k",
                                                    "vit_tiny_r_s16_p8_224_in21k",
                                                    "vit_small_patch16_224_in21k",
                                                    "vit_small_patch32_224_in21k",
                                                    "vit_base_patch16_224_in21k",
                                                    "vit_base_patch32_224_in21k",
                                                    "vit_large_patch16_224_in21k",
                                                    "vit_large_patch32_224_in21k"
                                                    ])

    train_name = trial.suggest_categorical("train", ["None", "Pretrained"])
    #train_name = trial.suggest_categorical("train", ["Pretrained"])

    if train_name == "None":
        pretrained = False
    if train_name == "Pretrained":
        pretrained = True
    
    #model = timm.create_model(model_name, img_size=[576,288], pretrained=pretrained, num_classes=2)
    
    Lossese = trial.suggest_categorical("Loss", ["CE", "Focal"])

    aug_color = trial.suggest_categorical("ColorJitter", ["None", "Color"])
    aug_Perspective = trial.suggest_categorical("Perspective", ["None", "Perspective"])
    aug_Rotation = trial.suggest_categorical("Rotation", ["None", "Rotation"])

    param_kwargs ={}

    if Lossese == "Focal":
        gamma_pram = trial.suggest_uniform('gamma', 0., 2.)
        param_kwargs["gamma_pram"]=gamma_pram

    all_acc = 0
    for split in range(5):
        print(split)
        acc = split_acc(split, aug_color, aug_Perspective, aug_Rotation, model_name, pretrained, Lossese, **param_kwargs)
        all_acc += acc
    best_acc = all_acc/5

    return -best_acc


############## main ##############
if __name__ == '__main__':
    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # 高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    TRIAL_SIZE = 200
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    with open(f"optuna_vit.txt", mode='w') as f:
        for key, value in study.best_params.items():
            f.write(f"{key}\t{value}\n")

        f.write(f"best acc:{-study.best_value}")