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
from models.Unet import UNet
from utils.Loss5 import SquareLoss


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
    train_transform1 = ut.Compose([#ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    #ut.RandomHorizontalFlip(p=0.5),   # ランダムに左右反転(p=probability)
                                    #ut.RandomVerticalFlip(p=0.5),
                                    #ut.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # 画像の明るさ、コントラスト、彩度、色相をランダムに変換
                                    #ut.GaussianBlur1(kernel_size=13, p=0.5),
                                    #ut.RandomGrayscale(p=0.1), # 確率pでグレースケールにランダムに変換
                                    ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    #ut.AddGaussianNoise(std=0.1, p=0.25),
                                    #ut.RandomErasing1(p=0.25), #四角形で画像をマスクする
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    train_transform2 = ut.Compose([#ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
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
    """
    train_dataset1 = Covid19_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform1)
    train_dataset2 = Covid19_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform2)
    val_dataset1 = Covid19_Loader(dataset_type='val', transform=val_transform)
    val_dataset2 = Covid19_Loader(dataset_type='val', transform=val_transform)
    """
    train_dataset1 = Cell_Dataset_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform1)
    train_dataset2 = Cell_Dataset_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform2)
    val_dataset1 = Cell_Dataset_Loader(dataset_type='val', transform=val_transform)
    val_dataset2 = Cell_Dataset_Loader(dataset_type='val', transform=val_transform)
    
    

    #これらをやると高速化、しかしHykwでは処理が重くなる
    #num_workers=os.cpu_count(),pin_memory=True
    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                                batch_size=args.batchsize,
                                                shuffle=True,
                                                drop_last=False)
    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                                batch_size=args.batchsize,
                                                shuffle=True,
                                                drop_last=False)
    
    val_loader1 = torch.utils.data.DataLoader(val_dataset1,
                                            batch_size=args.Tbatchsize,
                                            shuffle=False,
                                            drop_last=False)
    val_loader2 = torch.utils.data.DataLoader(val_dataset2,
                                            batch_size=args.Tbatchsize,
                                            shuffle=False,
                                            drop_last=False)


    return train_loader1, train_loader2, val_loader1, val_loader2


############## train関数 ##############
def train():
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss0 = 0 #class0のloss
    sum_loss1 = 0 #class1のloss
    sum_loss2 = 0 #class2のloss
    #sum_loss3 = 0 #class3のloss
    sum_loss4 = 0 #提案手法のloss
    sum_loss5 = 0 #softmax cross entropy loss
    

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    # 2枚の入力画像を準備する(同じ2枚でも違う2枚でもいい)
    for batch_idx, ((inputs1, targets1), (inputs2, targets2)) in tqdm(enumerate(zip(train_loader1,train_loader2)), total=len(train_loader1), leave=False):

        #GPUモード高速化
        inputs1 = inputs1.cuda(device, non_blocking=True)
        targets1 = targets1.cuda(device, non_blocking=True)
        inputs2 = inputs2.cuda(device, non_blocking=True)
        targets2 = targets2.cuda(device, non_blocking=True)  
        
        # 教師ラベルをlongモードに変換
        targets1 = targets1.long()
        targets2 = targets2.long()

        # 入力画像をモデルに入力
        #output = model(inputs)
        #y, h_0, h_1, h_2, h_3 = model(inputs1)
        #yy, h_00, h_11, h_22, h_33 = model(inputs2)
        y, h_0, h_1, h_2 = model(inputs1)
        yy, h_00, h_11, h_22 = model(inputs2)


        # 損失計算
        #loss = criterion(output, targets)
        #loss0, loss1, loss2, loss3, loss4 = criterion1(h_0, h_1, h_2, h_3, targets1, y, h_00, h_11, h_22, h_33, targets2, yy)
        loss0, loss1, loss2, loss4 = criterion1(h_0, h_1, h_2, targets1, y, h_00, h_11, h_22, targets2, yy)
        loss5 = criterion2(yy, targets2)

        #loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
        loss = loss0 + loss1 + loss2 + loss4 + loss5

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        #sum_loss += loss.item()
        sum_loss0 += loss0.item()
        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        #sum_loss3 += loss3.item()
        sum_loss4 += loss4.item()
        sum_loss5 += loss5.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()
    
    #return sum_loss / (batch_idx + 1)
    #return sum_loss0/(batch_idx+1), sum_loss1/(batch_idx+1), sum_loss2/(batch_idx+1), sum_loss3/(batch_idx+1), sum_loss4/(batch_idx+1), sum_loss5/(batch_idx+1)
    return sum_loss0/(batch_idx+1), sum_loss1/(batch_idx+1), sum_loss2/(batch_idx+1), sum_loss4/(batch_idx+1), sum_loss5/(batch_idx+1)


############## validation関数 ##############
def val():
    # モデル→推論モード
    model.eval()

    # 初期設定
    #sum_loss = 0
    sum_loss0 = 0
    sum_loss1 = 0
    sum_loss2 = 0
    #sum_loss3 = 0
    sum_loss4 = 0
    sum_loss5 = 0

    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, ((inputs1, targets1), (inputs2, targets2)) in tqdm(enumerate(zip(val_loader1, val_loader2)), total=len(val_loader1), leave=False):
            #GPUモード高速化
            inputs1 = inputs1.cuda(device, non_blocking=True)
            targets1 = targets1.cuda(device, non_blocking=True)
            inputs2 = inputs2.cuda(device, non_blocking=True)
            targets2 = targets2.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets1 = targets1.long()
            targets2 = targets2.long()

            # 入力画像をモデルに入力
            #output = model(inputs)
            #y, h_0, h_1, h_2, h_3 = model(inputs1)
            #yy, h_00, h_11, h_22, h_33 = model(inputs2)
            y, h_0, h_1, h_2 = model(inputs1)
            yy, h_00, h_11, h_22 = model(inputs2)

            # 損失計算
            #loss = criterion(output, targets)
            #loss0, loss1, loss2, loss3, loss4 = criterion1(h_0, h_1, h_2, h_3, targets1, y, h_00, h_11, h_22, h_33, targets2, yy)
            loss0, loss1, loss2, loss4 = criterion1(h_0, h_1, h_2, targets1, y, h_00, h_11, h_22, targets2, yy)
            loss5 = criterion2(yy, targets2)
            
            # loss溜め
            #sum_loss += loss.item()
            sum_loss0 += loss0.item()
            sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
            #sum_loss3 += loss3.item()
            sum_loss4 += loss4.item()
            sum_loss5 += loss5.item()

            ###精度の計算###
            IoU_calculator.update(yy, targets2, mode="softmax")
        
        IoU, mean_IoU = IoU_calculator()
    #return sum_loss / (batch_idx + 1), IoU, mean_IoU
    #return sum_loss0/(batch_idx+1), sum_loss1/(batch_idx+1), sum_loss2/(batch_idx+1), sum_loss3/(batch_idx+1), sum_loss4/(batch_idx+1), sum_loss5/(batch_idx+1), IoU, mean_IoU
    return sum_loss0/(batch_idx+1), sum_loss1/(batch_idx+1), sum_loss2/(batch_idx+1), sum_loss4/(batch_idx+1), sum_loss5/(batch_idx+1), IoU, mean_IoU  



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
    n_classes = 3   # class number

    # モデル設定
    model = UNet(in_ch=in_channel, n_class=n_classes).cuda(device)
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
    train_loader1, train_loader2, val_loader1, val_loader2 = dataload()

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
        train_loss0, train_loss1, train_loss2, train_loss4, train_loss5 = train()
            
        #val_loss0, val_loss1, val_loss2, val_loss3, val_loss4, val_loss5, val_IoU, val_mIoU = val()
        val_loss0, val_loss1, val_loss2, val_loss4, val_loss5, val_IoU, val_mIoU = val()

            
        ##### 結果表示 #####
        result_text = "Epoch:{:5d}/{:5d} Train Loss:{:.5f} val Loss:{:.5f} val mIoU:{:.3%}"
        result_text = result_text.format(epoch + 1,
                                        args.num_epochs,
                                        train_loss4,
                                        val_loss4,
                                        val_mIoU)
            
        print(result_text)

        ##### 出力結果を書き込み #####

        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch + 1, train_loss4))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.5f}\t{:.2%}\n".format(epoch + 1, val_loss4, val_mIoU))
            
                
            

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


