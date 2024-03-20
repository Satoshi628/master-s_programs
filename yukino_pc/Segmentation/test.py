#coding: utf-8
##### ライブラリ読み込み #####
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
from Net import SegNet
from dataset import Drosophila_Loader
############################



############## dataloader 関数##############
def dataload():
    # dataset = [image, label]
    # datatype = "train" or "val" or "test", transform = augmentation and preprocceing
    # val_area is for cross-validation
    test_dataset = Drosophila_Loader(rootdir="Dataset", val_area=args.split, split='test')

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.Tbatchsize, shuffle=False, drop_last=False)

    return test_loader



############## test関数 ##############
def test():
    # モデル→推論モード
    model.eval()

    # 初期設定
    correct = 0
    total = 0

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            # GPUモード  
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            #入力画像をモデルに入力
            output = model(inputs)

            ###精度の計算###

            # 出力をsoftmax関数に(0~1)
            output = F.softmax(output, dim=1)
            # 最大ベクトル
            _, predicted = output.max(1)
            # total = 正解, correct = 予測 
            total += (targets.size(0)*targets.size(1)*targets.size(2))
            correct += predicted.eq(targets).sum().item()  #predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse


    return correct/total



############## main ##############
if __name__ == '__main__':
    ##### コマンド設定 #####
    parser = argparse.ArgumentParser(description='SemanticSegmentation')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU id')
    # データのスプリット(split)指定
    parser.add_argument('--split', '-s', type=int, default=1,
                        help='Split of data to be used')
    # ミニバッチサイズ(teste)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4,
                        help='Number of images in each mini-batch')
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')

    args = parser.parse_args()
    ######################


    ##### Drosophila dataset #####
    classes = 5   # class number
    inputs_ch = 1  # input channel


    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ##### 保存ディレクトリ・ファイル #####

    PATH = "{}/predict.txt".format(args.out)

    with open(PATH, mode = 'w') as f:
        pass


    # モデル設定
    model = SegNet(in_ch=inputs_ch, n_class=classes).cuda(device)


    # データ読み込み+初期設定 
    test_loader = dataload()


    # 学習済みモデルのロード
    model_path = PATH ="{}/model.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))


    ##### test #####
    test_accuracy = test()


    ##### 結果表示 #####
    print("TestAccuracy==={:.2f}%".format(test_accuracy*100))


    ##### 出力結果を書き込み #####
    with open(PATH, mode = 'a') as f:
        f.write("{:.2f}\n".format(test_accuracy))


