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
from utils.dataset import Covid19_Loader
from utils.evaluation import IoU
from utils.make_pseudo import Make_Pseudo
import utils.utils as ut
#from utils.loss import FocalLoss, IoULoss
from utils.loss import IoULoss, one_hot_changer
from models.Unet import UNet
from utils.Loss4 import SquareLoss


# 初期値の乱数設定
rand_seed = 2
random.seed(rand_seed)  # python
np.random.seed(rand_seed)  # numpy
torch.manual_seed(rand_seed)  # pytorch
torch.backends.cudnn.benchmark = False  #cuDNNが決定的アルゴリズムを使用する
torch.backends.cudnn.deterministic = True  #決定論的アルゴリズムを使用する

#疑似ラベルのignore classを定義
#pillowはcv2はマイナスの値を画像として保存、読み込みできないため正の値
IGNORE = 211 #中途半端な値(素数)にしたほうがミスりにくい

############## dataloader関数##############
def dataload(pseudo_path=None):
    ### data augmentation + preprocceing ###
    train_transform = ut.Compose([#ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    #ut.RandomHorizontalFlip(p=0.5),   # ランダムに左右反転(p=probability)
                                    #ut.RandomVerticalFlip(p=0.5),
                                    ut.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                    ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.RandomErasing1(p=0.5), #四角形で画像をマスクする
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    train_dataset = Covid19_Loader(dataset_type='train', pseudo_path=pseudo_path, transform=train_transform)
    val_dataset = Covid19_Loader(dataset_type='val', transform=val_transform)
    pseudo_dataset = Covid19_Loader(dataset_type='pseudo', transform=val_transform)

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

    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset,
                                            batch_size=args.Tbatchsize,
                                            shuffle=False,
                                            drop_last=False)

    return train_loader, val_loader, pseudo_loader


############## train関数 ##############
def train():
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss0 = 0 #class0のloss
    sum_loss1 = 0 #class1のloss
    sum_loss2 = 0 #class2のloss
    sum_loss3 = 0 #class3のloss
    sum_loss4 = 0 #提案手法のloss
    sum_loss5 = 0 #softmax cross entropy loss
    

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        #output = model(inputs)
        y, h_0, h_1, h_2, h_3 = model(inputs)

        # 損失計算
        #loss = criterion(output, targets)
        loss0, loss1, loss2, loss3, loss4 = criterion1(h_0, h_1, h_2, h_3, targets, y)
        loss5 = criterion2(y, targets)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        #sum_loss += loss.item()
        sum_loss0 += loss0.item()
        sum_loss1 += loss1.item()
        sum_loss2 += loss2.item()
        sum_loss3 += loss3.item()
        sum_loss4 += loss4.item()
        sum_loss5 += loss5.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()
    
    #return sum_loss / (batch_idx + 1)
    return sum_loss0/(batch_idx+1), sum_loss1/(batch_idx+1), sum_loss2/(batch_idx+1), sum_loss3/(batch_idx+1), sum_loss4/(batch_idx+1), sum_loss5/(batch_idx+1)


############## validation関数 ##############
def val():
    # モデル→推論モード
    model.eval()

    # 初期設定
    #sum_loss = 0
    sum_loss0 = 0
    sum_loss1 = 0
    sum_loss2 = 0
    sum_loss3 = 0
    sum_loss4 = 0
    sum_loss5 = 0

    
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

            # 損失計算
            #loss = criterion(output, targets)
            loss0, loss1, loss2, loss3, loss4 = criterion1(h_0, h_1, h_2, h_3, targets, y)
            loss5 = criterion2(y, targets)
            
            # loss溜め
            #sum_loss += loss.item()
            sum_loss0 += loss0.item()
            sum_loss1 += loss1.item()
            sum_loss2 += loss2.item()
            sum_loss3 += loss3.item()
            sum_loss4 += loss4.item()
            sum_loss5 += loss5.item()

            ###精度の計算###
            IoU_calculator.update(y, targets, mode="softmax")
        
        IoU, mean_IoU = IoU_calculator()
    #return sum_loss / (batch_idx + 1), IoU, mean_IoU
    return sum_loss0/(batch_idx+1), sum_loss1/(batch_idx+1), sum_loss2/(batch_idx+1), sum_loss3/(batch_idx+1), sum_loss4/(batch_idx+1), sum_loss5/(batch_idx+1), IoU, mean_IoU 

############## Make Pseudo関数 ##############
def make_pseudo():

    total = 0
    correct = 0
    den = 0

    # モデル→推論モード
    model.eval()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(pseudo_loader), total=len(pseudo_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            #output = model(inputs)
            y, h_0, h_1, h_2, h_3 = model(inputs)

            # 出力をsoftmax
            output = F.softmax(y, dim=1)

            # 最大ベクトル
            value, predicted = output.max(1)  # value:確信度の最大値, pre:class番号
            
            # 擬似ラベルがついてないところはIGNORE
            predicted[value < args.certainty] = n_classes

            # one hot
            predicted = one_hot_changer(predicted, n_classes + 1, dim=1)
            targets = one_hot_changer(targets, n_classes + 1, dim=1)

            # total = 正解, correct = 予測 
            total += predicted.sum(dim=(0,2,3))
            correct += (predicted * targets).sum(dim=(0,2,3))
            

            # 擬似ラベル全体の割合
            den += (targets.size(0) * targets.size(2) * targets.size(3))
        
        
            #Pseudo_Maker(output)
            Pseudo_Maker(y)
    
    #画像番号初期化
    Pseudo_Maker.reset()

    precision = correct / (total + 1e-7)

    # 擬似ラベルの平均
    heikin = correct[:4].mean() / (total[:4].mean() + 1e-7) # 0から3までを持ってくる
    
    # IGNOREラベルの割合
    num = total[4] / (den + 1e-7)

    return precision[0].item(), precision[1].item(), precision[2].item(), precision[3].item(), heikin, num
    

############## Calculate Pseudo Accracy関数 ##############
def colc_pseudo_accracy(pseudo_path):
    pass


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
    parser.add_argument('--certainty', '-c', type=float, default=0.98,
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
    model = UNet(in_ch=in_channel, n_class=n_classes).cuda(device)
    #マルチGPU
    model = torch.nn.DataParallel(model) if args.multi_gpu else model

    #精度計算、疑似ラベル、クラス定義
    IoU_calculator = IoU(n_classes)
    Pseudo_Maker = Make_Pseudo(root_dir="{}/iteration0001".format(args.out),threshold=args.certainty)

    #損失関数設定
    #criterion = IoULoss(n_class)
    #criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    criterion1 = SquareLoss(device=device)  #提案手法のloss
    criterion2 = nn.CrossEntropyLoss(ignore_index=IGNORE)  #Softmax Cross Entropy Loss

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD
    """
    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True
    """
    best_IoU = 0.0

    epochs = [50, 10, 10]
    
    #何回疑似ラベルを作成し、学習するか
    for iter_ in range(1, args.num_iters + 1):
        if not os.path.exists(f"{args.out}/iteration{iter_:04}"):
            os.mkdir(f"{args.out}/iteration{iter_:04}")
        
        #疑似ラベルの作成ディレクトリ更新
        Pseudo_Maker.update_rood_dir(f"{args.out}/iteration{iter_:04}")
        
        PATH_1 = f"{args.out}/iteration{iter_:04}/train.txt"  #train
        PATH_2 = f"{args.out}/iteration{iter_:04}/validation.txt" #val
        PATH_3 = f"{args.out}/iteration{iter_:04}/pseudo.txt" #pseudo
        
        with open(PATH_1, mode='w') as f:
            f.write("epoch\ttrain loss\n")
        with open(PATH_2, mode='w') as f:
            f.write("epoch\tvalidation loss\tmIoU\n")
        with open(PATH_3, mode='w') as f:
            pass
        
        #1つ前のiterationファイルを確認し、ないならNone
        if os.path.exists(f"{args.out}/iteration{iter_-1:04}"):
            pseudo_path = f"{args.out}/iteration{iter_-1:04}"
        else:
            pseudo_path = None

        # データ読み込み+初期設定
        train_loader, val_loader, pseudo_loader = dataload(pseudo_path)

        """
        #スケジューラー定義
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=args.num_epochs,
                                                cycle_mult=1.0,
                                                max_lr=args.lr,
                                                min_lr=0,
                                                warmup_steps=0,
                                                gamma=1.0)
        """
        #best_IoU 初期化
        best_IoU = 0.0

        use_epochs = epochs[iter_ -1]
        print(use_epochs)

        for epoch in range(use_epochs):
            # train
            #train_loss = train()
            train_loss0, train_loss1, train_loss2, train_loss3, train_loss4, train_loss5 = train()
            # train
            #val_loss, val_IoU, val_mIoU = val()
            val_loss0, val_loss1, val_loss2, val_loss3, val_loss4, val_loss5, val_IoU, val_mIoU = val()

            
            ##### 結果表示 #####
            result_text = "iter:{:3d} Epoch:{:5d}/{:5d} Train Loss:{:.5f} val Loss:{:.5f} val mIoU:{:.3%}"
            result_text = result_text.format(iter_,
                                            epoch + 1,
                                            use_epochs,
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
                pre0, pre1, pre2, pre3, heikin, total = make_pseudo()

                with open(PATH_3, mode='w') as f:
                    f.write("{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t{:.2%}\t\n".format(pre0, pre1, pre2, pre3, heikin,total))



            #最初にやると一つずれるため、スケジューラーの更新は最後
            #scheduler.step()
        #end for epoch
        #pseudo_acc = colc_pseudo_accracy()

