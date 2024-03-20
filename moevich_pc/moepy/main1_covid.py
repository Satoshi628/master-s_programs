#擬似ラベルなしの場合
#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
import argparse

#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Covid19_Loader, Cell_Dataset_Loader
from utils.evaluation import IoU
from utils.make_pseudo import Make_Pseudo
import utils.utils as ut
#from utils.loss import FocalLoss, IoULoss
from utils.loss import IoULoss, one_hot_changer
from models.Unet import UNet_4
from utils.Loss4_covid import SquareLoss


# 初期値の乱数設定
rand_seed = 2
random.seed(rand_seed)  # python
np.random.seed(rand_seed)  # numpy
torch.manual_seed(rand_seed)  # pytorch
torch.backends.cudnn.benchmark = False  #cuDNNが決定的アルゴリズムを使用する
torch.backends.cudnn.deterministic = True  #決定論的アルゴリズムを使用する

############## dataloader関数##############
def dataload(pseudo_path=None):
    ### data augmentation + preprocceing ###
    train_transform = ut.Compose([#ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    #ut.RandomHorizontalFlip(p=0.5),   # ランダムに左右反転(p=probability)
                                    #ut.RandomVerticalFlip(p=0.5),
                                    #ut.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # 画像の明るさ、コントラスト、彩度、色相をランダムに変換
                                    #ut.GaussianBlur1(kernel_size=13, p=0.5),
                                    #ut.RandomGrayscale(p=0.1), # 確率pでグレースケールにランダムに変換
                                    ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    #ut.AddGaussianNoise(std=0.1, p=0.25),
                                    #ut.RandomErasing1(p=0.5), #四角形で画像をマスクする
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    train_dataset = Covid19_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform)
    val_dataset = Covid19_Loader(dataset_type='val', transform=val_transform)
    
    #train_dataset = Cell_Dataset_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform)
    #val_dataset = Cell_Dataset_Loader(dataset_type='val', transform=val_transform)
    
    

    #これらをやると高速化、しかしHykwでは処理が重くなる
    #num_workers=os.cpu_count(),pin_memory=True
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batchsize,
                                                shuffle=True,
                                                drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.Tbatchsize,
                                            shuffle=False,
                                            drop_last=False)


    return train_loader, val_loader


############## train関数 ##############
def train():
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)   
        """
        # 可視化するとき
        torchvision.utils.save_image(inputs,
                                     "image2.png",
                                     nrow=5,
                                     normalize=True)
        input()
        """
        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        #output = model(inputs)
        y, h_0, h_1, h_2, h_3 = model(inputs)
        #y, h_0, h_1, h_2 = model(inputs)

        # 損失計算
        #loss = criterion(output, targets)
        loss0, loss1, loss2, loss3, loss4 = criterion1(h_0, h_1, h_2, h_3, targets, y)
        #loss0, loss1, loss2, loss4 = criterion1(h_0, h_1, h_2, targets, y)
        loss5 = criterion2(y, targets)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
        #loss = loss0 + loss1 + loss2 + loss4 + loss5

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()
    
    return sum_loss/(batch_idx+1)


############## validation関数 ##############
def val():
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            #output = model(inputs)
            y, h_0, h_1, h_2, h_3 = model(inputs)
            #y, h_0, h_1, h_2 = model(inputs)


            # 損失計算
            #loss = criterion(output, targets)
            loss0, loss1, loss2, loss3, loss4 = criterion1(h_0, h_1, h_2, h_3, targets, y)
            #loss0, loss1, loss2, loss4 = criterion1(h_0, h_1, h_2, targets, y)
            loss5 = criterion2(y, targets)
            
            loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            IoU_calculator.update(y, targets, mode="softmax")
        
        IoU, mean_IoU = IoU_calculator()
    return sum_loss/(batch_idx+1), IoU, mean_IoU 



############## main ##############
if __name__ == '__main__':
    ##### コマンド設定 #####
    parser = argparse.ArgumentParser(description='SemanticSegmentation')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU id')
    # マルチGPU使用
    parser.add_argument('--multi_gpu', '-m', type=bool, default=False,
                        help='if True then use multi gpu')
    # ミニバッチサイズ(train)指定
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    # ミニバッチサイズ(validation)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4,
                        help='Number of images in each mini-batch')
    # 学習回数(epoch)指定
    parser.add_argument('--num_epochs', '-e', type=int, default=65536,
                        help='Number of epoch')
    # 反復回数(Iteration)指定
    parser.add_argument('--num_iters', '-i', type=int, default=3,
                        help='Number of iteration')
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')
    # 学習率指定
    parser.add_argument('--lr', '-l', type=float, default=1e-4,
                        help='Learning rate')
    # 疑似ラベル作成の閾値指定
    parser.add_argument('--certainty', '-c', type=float, default=0.95,
                        help='Certainty threshold')
    args = parser.parse_args()
    ######################

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))


    #入力チャンネル、出力クラス設定
    in_channel = 3  # input channnel(RGB)
    n_classes = 4   # class number

    # モデル設定
    model = UNet_4(in_ch=in_channel, n_class=n_classes).cuda(device)
    #マルチGPU
    model = torch.nn.DataParallel(model) if args.multi_gpu else model

    #精度計算、疑似ラベル、クラス定義
    IoU_calculator = IoU(n_classes)
    

    #損失関数設定
    criterion1 = SquareLoss(device=device)  #提案手法のloss
    criterion2 = nn.CrossEntropyLoss()  #Softmax Cross Entropy Loss

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD
    """
    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True
    """
    best_IoU = 0.0

    
    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))
        
        
    PATH_1 = "{}/train.txt".format(args.out)  #train
    PATH_2 = "{}/validation.txt".format(args.out) #val
        
    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tvalidation loss\tmIoU\n")
    


    # データ読み込み+初期設定
    train_loader, val_loader = dataload()

    """
    #スケジューラー定義
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                            first_cycle_steps=20000,
                                            cycle_mult=1.0,
                                            max_lr=args.lr,
                                            min_lr=0,
                                            warmup_steps=0,
                                            gamma=0.5)  #cosの高さ
    """

    #best_IoU 初期化
    best_IoU = 0.0
    start_time = time.time()


    for epoch in range(args.num_epochs):
    
        #train_loss0, train_loss1, train_loss2, train_loss3, train_loss4, train_loss5 = train()
        train_loss = train()
        
        #val_loss0, val_loss1, val_loss2, val_loss3, val_loss4, val_loss5, val_IoU, val_mIoU = val()
        val_loss, val_IoU, val_mIoU = val()

        
        ##### 結果表示 #####
        result_text = "Epoch:{:5d}/{:5d} Train Loss:{:.5f} val Loss:{:.5f} val mIoU:{:.3%}"
        result_text = result_text.format(epoch + 1,
                                        args.num_epochs,
                                        train_loss,
                                        val_loss,
                                        val_mIoU)
            
        print(result_text)

        ##### 出力結果を書き込み #####

        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch + 1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.5f}\t{:.2%}\n".format(epoch + 1, val_loss, val_mIoU))
            
                
            

        #モデル保存 & 疑似ラベル作成
        if val_mIoU > best_IoU:
            best_IoU = val_mIoU
            PATH = "{}/model.pth".format(args.out)
            torch.save(model.state_dict(), PATH)

            #最初にやると一つずれるため、スケジューラーの更新は最後
            #scheduler.step()
        #end for epoch
        #pseudo_acc = colc_pseudo_accracy()

    end_time = time.time()
    print("end_time", end_time - start_time)

