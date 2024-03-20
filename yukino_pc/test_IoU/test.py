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
############################

#正しいIoU値いじらないでください
correct_IoU = [0.9743227958679199,
        0.4915649890899658,
        0.5915558934211731,
        0.0443559102714061,
        0.5254498720169067]

#許容誤差 0.01%までは保障 もっと簡単にしてもいいです
error = 0.00005

def nearly_equal(iou0, iou1, iou2, iou3, miou):
    iou0_correct_flag = error > abs(iou0 - correct_IoU[0])
    iou1_correct_flag = error > abs(iou1 - correct_IoU[1])
    iou2_correct_flag = error > abs(iou2 - correct_IoU[2])
    iou3_correct_flag = error > abs(iou3 - correct_IoU[3])
    miou_correct_flag = error > abs(miou - correct_IoU[4])

    return iou0_correct_flag & iou1_correct_flag & iou2_correct_flag & iou3_correct_flag & miou_correct_flag

############## IoU test関数 ##############
def test():
    # モデル→推論モード
    #model.eval()

    # 初期設定
    #correct = 0
    #total = 0

    # IoU計算用データ読み込み
    targets_all = np.load("targets.npy")  #ラベル読み込み size=>[20, 256, 256]
    output_all = np.load("output.npy")  #モデルの出力値読み込み size=>[20, 4, 256, 256]
    
    # batch size 4固定
    targets_all = [torch.tensor(tar) for tar in  np.array_split(targets_all, 5)]
    output_all = [torch.tensor(out) for out in np.array_split(output_all, 5)]

    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for batch_idx, (targets, output) in enumerate(tqdm(zip(targets_all, output_all), leave=False)):
        # GPUモード  
        targets = targets.cuda(device)
        output = output.cuda(device)

        # 教師ラベルをlongモードに変換
        targets = targets.long()
        

        #入力画像をモデルに入力
        #output = model(inputs)

        #ここから先を編集して精度をだしてください
        #それぞれのクラスのIoUはiou0,iou1,iou2,iou3,miouにしてください
        ###精度の計算###
        


        
    #pythonのfloat型にしてください
    # torchを使っている場合 iou0 = iou0_tensor.item() で変換できます
    iou0 = 0.0
    iou1 = 0.0
    iou2 = 0.0
    iou3 = 0.0
    miou = 0.0
    
    #ここから先いじらない
    correct_flag = nearly_equal(iou0, iou1, iou2, iou3, miou)
    if correct_flag:
        print("合ってると思います")
    else:
        print("違うかも")
    print("IoU 精度")
    print("iou0".ljust(20) + f"{iou0:%}")
    print("iou1".ljust(20) + f"{iou1:%}")
    print("iou2".ljust(20) + f"{iou2:%}")
    print("iou3".ljust(20) + f"{iou3:%}")
    print("miou".ljust(20) + f"{miou:%}")





############## main ##############
if __name__ == '__main__':
    ##### コマンド設定 #####
    parser = argparse.ArgumentParser(description='SemanticSegmentation')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU id')
    args = parser.parse_args()
    ######################

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    ##### test #####
    test()
