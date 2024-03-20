#coding: utf-8
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
import timm


#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_OCT_VIT, Eye_Loader_OCT_VIT_SMOKE
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss
from utils.evaluation import Precition_Recall

############## dataloader関数##############
def dataload(split, age, sex, smoke, mask):
    ### data augmentation + preprocceing ###
    augment_list = [ndarrayToTensor(),
                    Resize([315,315]),
                    RandomCrop(size=(288, 288)),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    #ColorJitter(brightness=[0.0,1.0], contrast=[0.0,1.0], saturation=[0.0,1.0], hue=[0.0,0.5]),
                    #RandomPerspective(distortion_scale=0.5, p=0.5),
                    RandomRotation([-45.,45.]),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    train_transform = Compose(augment_list)

    val_transform = Compose([ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_dataset = Eye_Loader_OCT_VIT_SMOKE(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='train', split=split,
            use_age=age,
            use_sex=sex,
            use_smoke=smoke,
            use_mask=mask,
            transform=train_transform)
    val_dataset = Eye_Loader_OCT_VIT_SMOKE(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='val', split=split,
            use_age=age,
            use_sex=sex,
            use_smoke=smoke,
            use_mask=mask,
            transform=val_transform)
    test_dataset = Eye_Loader_OCT_VIT_SMOKE(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='test', split=split,
            use_age=age,
            use_sex=sex,
            use_smoke=smoke,
            use_mask=mask,
            transform=val_transform)
    
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
    
    acc_calc = Precition_Recall()

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, OCT, targets, info) in enumerate(train_loader):
        # GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        OCT = OCT.float().cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        for key, value in info.items():
            info[key] = value.cuda(device, non_blocking=True)

        # H方向に画像を結合
        inputs = torch.cat([inputs, OCT], dim=-2)
        #inputs = OCT

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output, attens = model(inputs, info)

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
    
    acc_calc = Precition_Recall()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, targets, info) in enumerate(val_loader):
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            for key, value in info.items():
                info[key] = value.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)
            #inputs = OCT

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs, info)

            ###精度の計算###
            acc_calc.update(output, targets)
    
    Accuracy, Precition, Recall = acc_calc()
    return Accuracy, Precition, Recall

############## val関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()
    acc_calc = Precition_Recall()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, targets, info) in enumerate(test_loader):
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            for key, value in info.items():
                info[key] = value.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)
            #inputs = OCT

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs, info)

            ###精度の計算###
            acc_calc.update(output, targets)
    
    Accuracy, Precition, Recall = acc_calc()
    return Accuracy, Precition, Recall

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    train_loader, val_loader, test_loader = dataload(cfg.dataset.split, cfg.dataset.use_age, cfg.dataset.use_sex, cfg.dataset.use_smoke, cfg.dataset.use_mask)

    with open("optuna_best.txt", mode='w') as f:
        f.write("ARMS2 Accuracy\tCFH Accuracy\tARMS2 Precition\tCFH Precition\tARMS2 Recall\tCFH Recall\n")

    device = torch.device('cuda:0')

    #model_name = "vit_tiny_patch16_224_in21k"
    model_name = "vit_large_patch32_224_in21k"
    pretrained = False
    model = timm.create_model(model_name, img_size=[576,288], pretrained=pretrained, num_classes=4,
        use_age=cfg.dataset.use_age,
        use_sex= cfg.dataset.use_sex,
        use_smoke= cfg.dataset.use_smoke)
    #model = timm.create_model(model_name, img_size=[288,288], pretrained=pretrained, num_classes=4)

    #gamma_pram = 0.9779477091945803
    gamma_pram = 0.9003473433504656
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
    # args.num_epochs = max epoch
    for epoch in range(20):
        # train
        train(model, train_loader, criterion, optimizer, device)
        # validation
        Accuracy, Precition, Recall = val(model, val_loader, device)
        print(f"{epoch+1}/{20}epoch: ARMS2 Accuracy:{Accuracy[0]:.2%} CFH Accuracy:{Accuracy[1]:.2%}")


        if Accuracy.mean() > best_acc:
            best_acc = Accuracy.mean()
            print(f"{epoch+1}/{20}epoch: best ARMS2 Accuracy:{Accuracy[0]:.2%} best CFH Accuracy:{Accuracy[1]:.2%}")
            best_Accuracy, best_Precition, best_Recall = test(model, test_loader, device)
            PATH = "model_val_acc.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)

        scheduler.step()
    print(best_Accuracy)
    with open(f"optuna_best.txt", mode='a') as f:
            f.write(f"{best_Accuracy[0]}\t{best_Accuracy[1]}\t{best_Precition[0]}\t{best_Precition[1]}\t{best_Recall[0]}\t{best_Recall[1]}\n")


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
