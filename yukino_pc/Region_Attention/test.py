#coding: utf-8
#----- 標準ライブラリ -----#
import matplotlib.pyplot as plt
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.dataset import Covid19_Loader, Drosophila_Loader
from utils.visualization import IoU_image
from utils.evaluation import IoU
import utils.utils as ut
from models.Unet import UNet

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    if cfg.dataset == "Covid19":
        test_dataset = Covid19_Loader(root_dir=cfg.Covid19.root_dir, dataset_type='test', transform=test_transform)

    if cfg.dataset == "Drosophila":
        test_dataset = Drosophila_Loader(rootdir=cfg.Drosophila.root_dir, val_area=cfg.Drosophila.val_area, split='test')

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.parameter.batch_size,
                                            shuffle=False,
                                            drop_last=False)

    # dataloader作成高速化(大規模データだったら有効)
    """
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.parameter.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=os.cpu_count(),
                                            pin_memory=True)
    """
    
    return test_loader


############## test関数 ##############
def test(model, test_loader,visual, IoU_calculator, device):
    # モデル→推論モード
    model.eval()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output = model(inputs)

            #可視化
            visual.save_fig(inputs, mode="input")
            visual.save_fig(targets, mode="label")
            visual.save_fig(output, mode="output")

            ###精度の計算###
            IoU_calculator.update(output, targets, mode="softmax")

        IoU, mean_IoU = IoU_calculator()
    return IoU, mean_IoU



@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")

    #dataset判定
    dataset = ["Covid19", "Drosophila"]
    if not cfg.dataset in ["Covid19", "Drosophila"]:
        raise ValueError("hydraのdatasetパラメータが違います.datasetは{}のみが許されます.".format(dataset))

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    #データセットの入力チャンネルとクラス数,セグメンテーション色を選択
    if cfg.dataset == "Covid19":
        in_channel = cfg.Covid19.input_channel
        n_class = cfg.Covid19.classes
        color = [[0, 0, 0],
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0]]
    if cfg.dataset == "Drosophila":
        in_channel = cfg.Drosophila.input_channel
        n_class = cfg.Drosophila.classes
        color = [[255, 255, 255],
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0],
                [0, 0, 0]]

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test.txt"
    with open(PATH, mode='w') as f:
        for i in range(n_class):
            f.write("class{} IoU\t".format(i))
        f.write("mIoU\n")

    # モデル設定
    model = UNet(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model
    
    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))


    #精度計算クラス定義
    IoU_calculator = IoU(n_class)

    #可視化クラス定義
    segmentation_visualer = IoU_image(color)

    # データ読み込み+初期設定
    test_loader = dataload(cfg)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    IoU_acc, mean_IoU = test(model, test_loader,segmentation_visualer, IoU_calculator, device)

    with open(PATH, mode='a') as f:
        for class_iou in IoU_acc:
            f.write("{:.3%}\t".format(class_iou))
        f.write("{:.3%}\n".format(mean_IoU))
    
    #Accuracy表示
    print("IoU")
    for idx, class_iou in enumerate(IoU_acc):
        print("class{} IoU      :{:.3%}".format(idx, class_iou))
    
    print("mean IoU         :{:.3%}".format(mean_IoU))



############## main ##############
if __name__ == '__main__':
    main()
