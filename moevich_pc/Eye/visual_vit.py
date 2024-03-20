#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
import re

#----- 専用ライブラリ -----#
import pandas as pd
import glob
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from PIL import Image
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
from timm_vis import methods

#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_ARMS2, Eye_Loader_OCT_VIT
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss


############## dataloader関数##############
def dataload():
    val_transform = Compose([ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    test_dataset = Eye_Loader_OCT_VIT(root_dir="/mnt/hdd1/kamiya/dataset/Eye", dataset_type='test', split=0, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    # これらをやると
    # num_workers=os.cpu_count(),pin_memory=True
    return test_loader



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
            #OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            #inputs = torch.cat([inputs, OCT], dim=-2)

            # 教師ラベルをlongモードに変換
            targets = targets.long()
            print(model)

            methods.grad_cam(model,model.blocks[11])
    
    return (correct/total).item()

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    device = torch.device('cuda:0')

    test_loader = dataload()


    model = timm.create_model('vit_base_patch16_224_in21k', img_size=[576,288], pretrained=True, num_classes=2)
    #model = timm.create_model('vit_base_patch16_224_in21k', img_size=[288,288], pretrained=True, num_classes=2)
    
    model_path = "model.pth"
    model.load_state_dict(torch.load(model_path))

    model = model.cuda(device)
    image_paths = sorted(glob.glob(os.path.join("/mnt/hdd1/kamiya/dataset/Eye","images","*")))
    info = pd.read_csv(os.path.join("/mnt/hdd1/kamiya/dataset/Eye", "info.csv"))

    image_num = [int(re.search(r"case\d+",path).group()[4:]) for path in image_paths]
    image_num = np.array(image_num)

    label = [info.query(f"AI_number == {num}").iloc[:,-2:].values for num in image_num]
    label = np.array(label)[:,0,0]
    #methods.grad_cam(model,model.blocks[11],image_paths[0],0, save_path="grad_cam.png")
    """
    for i in range(len(image_paths)):
        img = Image.open(image_paths[i])
        img.save(f"img{i}_label{label[i]}.png")
        methods.maximally_activated_patches_img(model,image_paths[i], save_path=f"active_patche{i}_label{label[i]}.png")
        #methods.saliency_map(model,image_paths[i], save_path=f"activations{i}.png")
        if i == 6:
            break
    """
    for i, (inputs, OCT, targets) in enumerate(test_loader):
        inputs = inputs.cuda(device, non_blocking=True)
        OCT = OCT.float().cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # H方向に画像を結合
        inputs = torch.cat([inputs, OCT], dim=-2)

        # 教師ラベルをlongモードに変換
        targets = targets.long()
        label_list = targets.tolist()
        mean, std = torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])
        mean, std = mean.to(device)[None,:,None,None], std.to(device)[None,:,None,None]
        for batch_idx in range(inputs.shape[0]):
            img = inputs[batch_idx][None]
            show_img = img*std+mean
            torchvision.utils.save_image(show_img,f"img{i*inputs.shape[0]+batch_idx}_label{label_list[batch_idx]}.png")
            methods.maximally_activated_patches_img(model,img, save_path=f"active_patche{i*inputs.shape[0]+batch_idx}_label{label_list[batch_idx]}.png")

        break

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
