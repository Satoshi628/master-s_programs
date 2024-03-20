#coding: utf-8
##### ライブラリ読み込み #####
import _common_function as cf
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import argparse
import random
from tqdm import tqdm
from Net_AML_unet_FlyCell import Generator, Discriminator
from dataset import FlyCellDataLoader_crossval
import utils as ut
import time
import sys
# sys.path.append('../')
############################


############## dataloader 関数##############
def dataload():
    ds_test = FlyCellDataLoader_crossval(
        rootdir=args.rootdir, val_area=args.val_area, split='test')

    test_loader = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batchsize, shuffle=False, num_workers=args.threads)
    return test_loader

############## test関数 ##############


def test():
    # モデル→推論モード
    model_G.eval()

    # 初期設定
    correct = 0
    total = 0
    Intersection = 0
    union = 0

    start_time = time.time()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            # GPUモード
            #inputs = inputs.cuda(device)
            #targets = targets.cuda(device)

            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            #output = model(inputs)
            """val_img = model_G(inputs_a, memory=None, Train=False)

            DI_out = model_D(inputs_a, val_img, None, None, None)

            val_img = model_G(inputs_a, DI_out[1:], Train=False)
            """
            # 入力画像をGeneratorに入力
            output = model_G(inputs, memory=None, Train=False)

            DI_out = model_D(inputs, output, None, None, None)

            output = model_G(inputs, DI_out[1:], Train=False)

            # print(output)
            # print(output[0].shape)
            #a = torch.max(output[0])
            #b = torch.min(output[0])
            #print('Max', a)
            #print('Min', b)
            # input()

            # 画像保存

            # print(inputs.type)  # Tensor
            # output:'tuple' object has no attribute 'type' /output[0] : tensor !!
            # print(output[0].type)
            # print(targets.type)  # Tensor
            # input()
            image.save_Segmentation(
                inputs[:4], output[:4], targets[:4], folder=args.out, mode='sigmoid')
            image.save_Segmentation(
                inputs[4:8], output[4:8], targets[4:8], folder=args.out, mode='sigmoid')
            image.save_Segmentation(
                inputs[8:12], output[8:12], targets[8:12], folder=args.out, mode='sigmoid')
            image.save_Segmentation(
                inputs[12:16], output[12:16], targets[12:16], folder=args.out, mode='sigmoid')

            ###精度の計算###
            # 出力をsoftmax関数に(0~1)
            #output = F.softmax(output, dim=1)
            output = torch.sigmoid(output)

            # 最大ベクトル
            _, predicted = output.max(1)

            # total = 正解, correct = 予測
            total += (targets.size(0)*targets.size(1)*targets.size(2))
            # predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse
            correct += predicted.eq(targets).sum().item()

            Intersection += torch.stack([(predicted.eq(i)
                                        & targets.eq(i)).sum() for i in range(5)])
            union += torch.stack([(predicted.eq(i) | targets.eq(i)).sum()
                                 for i in range(5)])

        IoU = (Intersection / (union + 1e-24)).float()
        mean_IoU = IoU.mean()

        end_time = time.time()
        print('Test_Time_in_sec:', (end_time - start_time))

    return correct / total, IoU, mean_IoU


############## main ##############
if __name__ == '__main__':
    ##### コマンド設定 #####
    parser = argparse.ArgumentParser(description='FlyCell')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU id')

    # ミニバッチサイズ(学習)指定
    parser.add_argument('--batchsize', '-b', type=int, default=32)

    # ミニバッチサイズ(test)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4,
                        help='Number of images in each mini-batch')
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')

    parser.add_argument("--rootdir", type=str,
                        default='/mnt/addhdd/Usha/AML_unet_FlyCell/data')

    # iter : イテレーション数(1epoch内で何回ランダムクロップを繰り返すか)
    parser.add_argument("--iter", default=12, type=int)

    #threads : num_workers数;何個GPUを使っているか
    parser.add_argument("--threads", default=2, type=int)

    # val_area : validationのブロック番号(１～５を選択). これを変えることで5回のCross Validationが可能
    parser.add_argument("--val_area", '-v', type=int, default=1,
                        help='cross-val test area [default: 5]')

    args = parser.parse_args()
    ######################

    ##### Covid-19 dataset #####

    classes = 5  # class number
    inputs_ch = 1  # input channel

    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    ##### 保存ディレクトリ・ファイル #####

    PATH = "{}/test_result.txt".format(args.out)

    with open(PATH, mode='w') as f:
        pass

    # モデル設定
    model_G = Generator(n_channels=inputs_ch, n_classes=classes, device=device,
                        bilinear=True).cuda(device)
    model_D = Discriminator(in_ch=inputs_ch + classes, out_ch=5).cuda(device)

    # 複数GPU使用
    model_G = torch.nn.DataParallel(model_G)
    model_D = torch.nn.DataParallel(model_D)

    # 高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    # image保存クラス
    color = [[0, 0, 0],
             [0, 0, 255],
             [0, 255, 0],
             [255, 0, 0],
             [139, 194, 136]]
    image = cf.Image(color=color)

    # データ読み込み+初期設定
    test_loader = dataload()

    # 学習済みモデルのロード
    model_path = "{}/model_G_IoU.pth".format(args.out)
    model_G.load_state_dict(torch.load(model_path))

    print("mIoUが最高精度のときの精度")
    ##### test #####
    test_accuracy, test_IoU_iou, test_mIoU_iou = test()

    ##### 結果表示 #####
    text1 = "TestAccuracy==={:.2f}% TestmIoU==={:.2f}%".format(
        test_accuracy * 100,
        test_mIoU_iou * 100
    )

    text2 = "IoU[0]=={:.2f}% IoU[1]=={:.2f}% IoU[2]=={:.2f}% IoU[3]=={:.2f}% IoU[4]=={:.2f}%".format(
        test_IoU_iou[0] * 100,
        test_IoU_iou[1] * 100,
        test_IoU_iou[2] * 100,
        test_IoU_iou[3] * 100,
        test_IoU_iou[4] * 100
    )
    print(text1)
    print(text2)

    # 学習済みモデルのロード
    model_path = "{}/model_G_loss.pth".format(args.out)
    model_G.load_state_dict(torch.load(model_path))

    print("lossが最低のときの精度")
    ##### test #####
    test_accuracy, test_IoU_loss, test_mIoU_loss = test()

    ##### 結果表示 #####
    text1 = "TestAccuracy==={:.2f}% TestmIoU==={:.2f}%".format(
        test_accuracy * 100,
        test_mIoU_loss * 100
    )

    text2 = "IoU[0]=={:.2f}% IoU[1]=={:.2f}% IoU[2]=={:.2f}% IoU[3]=={:.2f}% IoU[4]=={:.2f}%".format(
        test_IoU_loss[0] * 100,
        test_IoU_loss[1] * 100,
        test_IoU_loss[2] * 100,
        test_IoU_loss[3] * 100,
        test_IoU_loss[4] * 100
    )
    print(text1)
    print(text2)

    ##### 出力結果を書き込み #####
    if test_mIoU_iou > test_mIoU_loss:
        with open(PATH, mode='a') as f:
            f.write("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(test_IoU_iou[0] * 100,
                                                                  test_IoU_iou[1] * 100,
                                                                  test_IoU_iou[2] * 100,
                                                                  test_IoU_iou[3] * 100,
                                                                  test_IoU_iou[4] * 100,
                                                                  test_mIoU_iou * 100))
    else:
        with open(PATH, mode='a') as f:
            f.write("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(test_IoU_loss[0] * 100,
                                                                  test_IoU_loss[1] * 100,
                                                                  test_IoU_loss[2] * 100,
                                                                  test_IoU_loss[3] * 100,
                                                                  test_IoU_loss[4] * 100,
                                                                  test_mIoU_loss * 100))
