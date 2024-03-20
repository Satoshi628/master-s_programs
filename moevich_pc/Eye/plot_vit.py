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
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_OCT_VIT, Eye_Loader_OCT_VIT_SMOKE
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss


############## dataloader関数##############
def dataload(split, age, sex, smoke, mask):
    ### data augmentation + preprocceing ###
    val_transform = Compose([ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    test_dataset = Eye_Loader_OCT_VIT_SMOKE(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='test', split=split,
            use_age=age,
            use_sex=sex,
            use_smoke=smoke,
            use_mask=mask,
            transform=val_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            drop_last=False)

    # これらをやると
    # num_workers=os.cpu_count(),pin_memory=True
    return test_loader


############## atention rollout関数 ##############
def attention_rollout(model, test_loader, device):
    # モデル→推論モード
    model.eval()
    if not os.path.exists(f"rollout"):
                os.mkdir(f"rollout")

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, targets, info) in enumerate(test_loader):
            #print(batch_idx)
            if batch_idx > 50:
                assert False
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)
            image = inputs[0].cpu().permute(1, 2, 0).numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image*std[None,None]+mean[None,None]
            image = (image*255).astype(np.uint8)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs)

            attens = torch.cat(attens, dim=0)
            attens = attens.sum(dim=1)
            #attens = attens[0]
            attens = attens + torch.eye(attens.shape[-1], device=attens.device)[None]
            #attens = attens/attens.sum(dim=(-1,-2),keepdim=True)
            v = attens[-1]
            for idx in range(1, attens.shape[0]):
                v = torch.matmul(v, attens[-1-idx])
            #mask  = v[0,1:].view(36,18).detach().to("cpu").numpy()
            mask  = v[0,1:].view(18,9).detach().to("cpu").numpy()
            #input(mask)
            attention_map = cv2.resize((mask-mask.min())/(mask.max()-mask.min()), [image.shape[1], image.shape[0]])
            alpha = 0.5
            colormap = plt.get_cmap('jet')
            heatmap = (colormap(attention_map) * 255).astype(np.uint16)[:, :, :3]
            # heatmap = heatmap[:,:,[2,1,0]]
            
            blended = image * alpha + heatmap * (1 - alpha)
            blended = blended.astype(np.uint8)
            blended = np.concatenate([image, blended],axis=1)
            plt.imshow(blended)
            plt.axis('off')
            plt.savefig(f"rollout/{batch_idx:04}.png")
            plt.cla()


############## val関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, targets, info) in enumerate(test_loader):
            if batch_idx > 50:
                assert False
            if not os.path.exists(f"OCT{batch_idx}"):
                os.mkdir(f"OCT{batch_idx}")
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)
            patches = inputs.unfold(2, 64, 64).unfold(3, 64, 64)
            patches = patches.flatten(2, 3)
            patches = patches.transpose(1, 2)
            image = inputs[0].cpu().permute(1, 2, 0).numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image*std[None,None]+mean[None,None]
            image = (image*255).astype(np.uint8)
            plt.imshow(image)
            plt.axis('off')
            plt.savefig(f"OCT{batch_idx}/image.png")
            plt.cla()

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs)
            attens = torch.stack(attens, dim=1)

            for patch, atten in zip(patches, attens):
                atten = F.avg_pool2d(atten, 4)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                patch = patch
                patch = patch.cpu().permute(0, 2, 3, 1).numpy()
                patch = patch*std[None,None,None]+mean[None,None,None]
                patch = (patch*255).astype(np.uint8)
                atten = atten.mean(dim=1).cpu().numpy()
                
                row = 2
                col = patches.shape[1] // row
                plt.figure(figsize=(10,3))

                num = 0

                while num < row * col:
                    num += 1
                    plt.subplot(row, col, num)
                    plt.imshow(patch[num-1])
                    plt.axis('off')
                plt.savefig(f"OCT{batch_idx}/yoko.png")
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                plt.cla()

                for idx, at in enumerate(atten):
                    plt.figure(figsize=(10,10))
                    plt.imshow(at, cmap='magma', interpolation='nearest')
                    plt.axis('off')
                    plt.savefig(f"OCT{batch_idx}/attention map_{idx}.png")
                    plt.cla()

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    test_loader = dataload(cfg.dataset.split, cfg.dataset.use_age, cfg.dataset.use_sex, cfg.dataset.use_smoke, cfg.dataset.use_mask)

    device = torch.device('cuda:0')

    #model_name = "vit_tiny_patch16_224_in21k"
    model_name = "vit_large_patch32_224_in21k"
    pretrained = True
    model = timm.create_model(model_name, img_size=[576,288], pretrained=pretrained, num_classes=4)
    model_path = "model_val_acc.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.cuda(device)
    attention_rollout(model, test_loader, device)
    test(model, test_loader, device)



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
