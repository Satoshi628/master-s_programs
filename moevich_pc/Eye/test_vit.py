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
from utils.dataset import Eye_Loader_OCT_VIT
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss


############## dataloader関数##############
def dataload(split):
    ### data augmentation + preprocceing ###
    augment_list = [ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]

    train_transform = Compose(augment_list)

    val_transform = Compose([ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_dataset = Eye_Loader_OCT_VIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='train', split=split, transform=train_transform)
    val_dataset = Eye_Loader_OCT_VIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='val', split=split, transform=val_transform)
    test_dataset = Eye_Loader_OCT_VIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='test', split=split, transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=8,
                                                shuffle=False,
                                                drop_last=False,
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


############## val関数 ##############
def test(model, train_loader, val_loader, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    useOCT_total = 0
    useOCT_correct = 0
    noOCT_total = 0
    noOCT_correct = 0
    total = 0
    correct = 0

    # 設定：パラメータの更新なし
    with torch.no_grad():
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

            ###精度の計算###
            total += output.size(0)
            correct += (targets == output.max(dim=-1)[1]).sum()

            OCT_nouse_flag = [torch.unique(OCT.flatten(1)[i],dim=-1).shape[0] for i in range(OCT.shape[0])]
            OCT_nouse_flag = torch.tensor(OCT_nouse_flag,device=OCT.device) == 3

            useOCT_total += (~OCT_nouse_flag).sum()
            useOCT_correct += (~OCT_nouse_flag*(targets == output.max(dim=-1)[1])).sum()

            noOCT_total += OCT_nouse_flag.sum()
            noOCT_correct += (OCT_nouse_flag*(targets == output.max(dim=-1)[1])).sum()

        for batch_idx, (inputs, OCT, targets) in enumerate(val_loader):
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

            ###精度の計算###
            total += output.size(0)
            correct += (targets == output.max(dim=-1)[1]).sum()

            OCT_nouse_flag = [torch.unique(OCT.flatten(1)[i],dim=-1).shape[0] for i in range(OCT.shape[0])]
            OCT_nouse_flag = torch.tensor(OCT_nouse_flag, device=OCT.device) == 3

            useOCT_total += (~OCT_nouse_flag).sum()
            useOCT_correct += (~OCT_nouse_flag*(targets == output.max(dim=-1)[1])).sum()

            noOCT_total += OCT_nouse_flag.sum()
            noOCT_correct += (OCT_nouse_flag*(targets == output.max(dim=-1)[1])).sum()

        for batch_idx, (inputs, OCT, targets) in enumerate(test_loader):
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

            ###精度の計算###
            total += output.size(0)
            correct += (targets == output.max(dim=-1)[1]).sum()

            OCT_nouse_flag = [torch.unique(OCT.flatten(1)[i],dim=-1).shape[0] for i in range(OCT.shape[0])]
            OCT_nouse_flag = torch.tensor(OCT_nouse_flag,device=OCT.device) == 3

            useOCT_total += (~OCT_nouse_flag).sum()
            useOCT_correct += (~OCT_nouse_flag*(targets == output.max(dim=-1)[1])).sum()

            noOCT_total += OCT_nouse_flag.sum()
            noOCT_correct += (OCT_nouse_flag*(targets == output.max(dim=-1)[1])).sum()
            
    print(correct)
    print(total)
    print(useOCT_correct)
    print(useOCT_total)
    print(noOCT_correct)
    print(noOCT_total)
    return (correct/total).item(), (useOCT_correct/useOCT_total).item(), (noOCT_correct/noOCT_total).item()

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    train_loader, val_loader, test_loader = dataload(cfg.dataset.split)

    device = torch.device('cuda:0')

    model_name = "vit_tiny_patch16_224_in21k"
    pretrained = False
    model = timm.create_model(model_name, img_size=[576,288], pretrained=pretrained, num_classes=2)
    model_path = "model_val_acc.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.cuda(device)
    all_acc, use_OCT_acc, nouse_OCT_acc = test(model, train_loader, val_loader, test_loader, device)
    print(f"All ACC:{all_acc:.2%}")
    print(f"use ACC:{use_OCT_acc:.2%}")
    print(f"no use ACC:{nouse_OCT_acc:.2%}")


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
