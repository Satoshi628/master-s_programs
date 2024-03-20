#coding: utf-8
#----- 標準ライブラリ -----#
import os
import argparse
import random
import sys
import time

#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#----- 自作モジュール -----#
from utils.evaluation import one_hot_changer, IoU
from utils.visualization import IoU_image
from models.AML import AML
from utils.dataset import FlyCellDataLoader_crossval


############## dataloader 関数##############
def dataload():
    ds_test = FlyCellDataLoader_crossval(
        rootdir=args.rootdir, val_area=args.val_area, split='test')

    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batchsize, shuffle=False, num_workers=args.threads)
    return test_loader


############## test関数 ##############
def test():
    # モデル→学習モード
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            #longモード
            targets_long = targets.long()
            #one hot vector化
            targets = one_hot_changer(targets_long, classes, dim=1)

            #outputs = model(inputs, targets)
            outputs = model(inputs)

            #可視化
            visual.save_fig(inputs, mode="input")
            visual.save_fig(targets_long, mode="label")
            visual.save_fig(outputs, mode="output")

            ###精度の計算###
            IoU_fnc.update(outputs, targets_long, mode='linear')

        iou, miou = IoU_fnc()

    return iou, miou


############## main ##############
if __name__ == '__main__':

    ####### コマンド設定 #######
    parser = argparse.ArgumentParser(description='FlyCell')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=0)
    # ミニバッチサイズ指定
    parser.add_argument('--batchsize', '-b', type=int, default=8)

    #データセットのディレクトリパス
    parser.add_argument("--rootdir", type=str,
                        default='/media/hykw/data/Kamiya/dataset/drosophila/data')

    # iter : イテレーション数(1epoch内で何回ランダムクロップを繰り返すか)
    parser.add_argument("--iter", default=12, type=int)

    #threads : num_workers数;何個GPUを使っているか
    parser.add_argument("--threads", default=2, type=int)

    # val_area : validationのブロック番号(１～５を選択). これを変えることで5回のCross Validationが可能
    parser.add_argument("--val_area", '-v', type=int, default=1,
                        help='cross-val test area [default: 5]')

    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result')

    args = parser.parse_args()


    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    ##### 保存ディレクトリ・ファイル #####
    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))
    

    PATH = "{}/test.txt".format(args.out)

    with open(PATH, mode='w') as f:
        f.write("class0 IoU\tclass1 IoU\tclass2 IoU\tclass3 IoU\tclass4 IoU\tmIoU\n")

    ##### drosophila dataset #####

    classes = 5   # class number
    in_channels = 1  # input channel

    # モデル設定
    model = AML(in_channels, classes, channel=32, class_weights=None).cuda(device)

    # 学習済みモデルのロード
    model_path = "{}/model.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))

    # IoU計算クラス定義
    IoU_fnc = IoU(classes)

    #可視化クラス定義
    color = [[255, 255, 255],
            [0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [0, 0, 0]]
    
    visual = IoU_image(root_dir="{}/images".format(args.out),color=color)


    # データ読み込み
    test_loader = dataload()

    torch.backends.cudnn.benchmark = True

    IoU_acc, mean_IoU = test()

    with open(PATH, mode='a') as f:
        for class_iou in IoU_acc:
            f.write("{:.3%}\t".format(class_iou))
        f.write("{:.3%}\n".format(mean_IoU))
    
    #Accuracy表示
    print("IoU")
    for idx, class_iou in enumerate(IoU_acc):
        print("class{} IoU      :{:.3%}".format(idx, class_iou))
    print("mean IoU         :{:.3%}".format(mean_IoU))

