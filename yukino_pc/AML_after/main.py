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
from models.AML import AML
from utils.loss import CrossEntropy
from utils.dataset import FlyCellDataLoader_crossval


############## dataloader関数##############
def dataload():
    ds_train = FlyCellDataLoader_crossval(
        rootdir=args.rootdir, val_area=args.val_area, split='train', iteration_number=args.batchsize*args.iter)
    ds_val = FlyCellDataLoader_crossval(
        rootdir=args.rootdir, val_area=args.val_area, split='val')

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batchsize, shuffle=True, num_workers=args.threads)
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batchsize, shuffle=False, num_workers=args.threads)

    return train_loader, val_loader


############## train関数 ##############
def train(epoch):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss_g = 0
    sum_loss_d = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
        # GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        #longモード
        targets_long = targets.long()
        #one hot vector化
        targets = one_hot_changer(targets_long, classes, dim=1)

        #Generatorを学習
        segmentation, fake_D, weights = model.learn_G(inputs, targets)
        
        zeros = torch.zeros_like(fake_D)

        ### 損失計算 ###
        #GT_PDA1,2,3+segmentation Cross Entropy Loss
        loss_G = model.get_G_Loss()

        loss_G_ad = lamda * criterion_ad(fake_D, zeros)
        loss_G = loss_G_ad + loss_G

        # 勾配を0
        model.zero_grad()

        # 誤差逆伝搬
        loss_G.backward()

        # パラメーター更新
        optimizer.step()

        #イニシャルのDはdetachしたことを表す
        D_segmentation = segmentation.detach()
        D_weights = [weight.detach() for weight in weights]

        #Discriminatorを学習
        real_D = model.learn_D(inputs, targets, D_weights)
        fake_D = model.learn_D(inputs, D_segmentation, D_weights)

        ones = torch.ones_like(real_D)
        zeros = torch.zeros_like(fake_D)

        ### 損失計算 ###
        loss_D_real = criterion_ad(real_D, ones)  # adversarial loss
        loss_D_fake = criterion_ad(fake_D, zeros)  # adversarial loss
        loss_D = (loss_D_real + loss_D_fake) / 2.0

        # 勾配を0
        model.zero_grad()

        # 誤差逆伝搬
        loss_D.backward()

        # パラメーター更新
        optimizer.step()

        # loss溜め
        sum_loss_g += loss_G.item()
        sum_loss_d += loss_D.item()
        del loss_G
        del loss_D

        IoU_fnc.update(D_segmentation, targets_long, mode='linear')

    iou, miou = IoU_fnc()

    return sum_loss_g / (batch_idx + 1), sum_loss_d / (batch_idx + 1), iou, miou


############## val関数 ##############
def val(epoch):
    # モデル→学習モード
    model.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            #longモード
            targets_long = targets.long()
            #one hot vector化
            targets = one_hot_changer(targets_long, classes, dim=1)

            outputs = model(inputs)

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
    # ミニバッチサイズ(学習)指定
    parser.add_argument('--batchsize', '-b', type=int, default=8)
    # ミニバッチサイズ(学習)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4)

    parser.add_argument("--rootdir", type=str,
                        default='/media/hykw/data/Kamiya/dataset/drosophila/data')

    # iter : イテレーション数(1epoch内で何回ランダムクロップを繰り返すか)
    parser.add_argument("--iter", default=12, type=int)

    #threads : num_workers数;何個GPUを使っているか
    parser.add_argument("--threads", default=2, type=int)

    # val_area : validationのブロック番号(１～５を選択). これを変えることで5回のCross Validationが可能
    parser.add_argument("--val_area", '-v', type=int, default=1,
                        help='cross-val test area [default: 5]')

    # 学習回数(epoch)指定
    parser.add_argument('--num_epochs', '-e', type=int, default=1000)
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result')
    # 乱数指定
    parser.add_argument('--seed', '-s', type=int, default=0)

    args = parser.parse_args()

    ##### 初期設定表示 #####

    print("[Experimental conditions]")
    print(" GPU ID         : {}".format(args.gpu))
    print(" Epochs         : {}".format(args.num_epochs))
    print(" Minibatch size : {}".format(args.batchsize))
    print("")

    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    ##### 保存ディレクトリ・ファイル #####
    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))

    PATH_1 = "{}/train.txt".format(args.out)
    PATH_2 = "{}/val.txt".format(args.out)

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss G\ttrain loss D\ttrain mIoU\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tval mIoU\n")

    ##### drosophila dataset #####

    classes = 5   # class number
    in_channels = 1  # input channel

    # モデル設定
    model = AML(in_channels, classes, channel=32, class_weights=None).cuda(device)

    # IoUクラス定義
    IoU_fnc = IoU(classes)


    # 損失関数設定
    lamda = 0.01
    criterion_ad = nn.BCEWithLogitsLoss(weight=None)

    # optimizer設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # 初期値の乱数設定
    random.seed(11)  # python
    np.random.seed(11)  # numpy
    torch.manual_seed(11)  # pytorch

    # データ読み込み
    train_loader, val_loader = dataload()

    torch.backends.cudnn.benchmark = True

    ##### training & validation #####
    best_miou = 0.

    for epoch in range(args.num_epochs):

        train_loss_g, train_loss_d, _, train_miou = train(epoch)  # train
        _, val_miou = val(epoch) #val

        ##### 結果表示 #####
        text = "Epoch{:3d}/{:3d}  LossG={:.4f}  LossD={:.4f} train_mIoU={:.3%}  val_mIoU={:.3%}"
        print(text.format(epoch + 1,
                        args.num_epochs,
                        train_loss_g,
                        train_loss_d,
                        train_miou,
                        val_miou))

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.4f}\t{:.3%}\n".format(epoch + 1, train_loss_g, train_loss_d, train_miou))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.3%}\n".format(epoch+1, val_miou))

        if val_miou > best_miou:
            best_miou = val_miou
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)
        
    print("最高mIoU:{:.3%}".format(best_miou))
