#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
import argparse

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Covid19_Loader
from utils.evaluation import IoU, Pseudo_accracy
from utils.make_pseudo import Make_Pseudo
import utils.utils as ut
from utils.loss import FocalLoss, IoULoss
from models.Unet import UNet

#疑似ラベルのignore classを定義
#pillowはcv2はマイナスの値を画像として保存、読み込みできないため正の値
IGNORE = 211 #中途半端な値(素数)にしたほうがミスりにくい

############## dataloader関数##############
def dataload(cfg, pseudo_path=None):
    ### data augmentation + preprocceing ###
    train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                    # ランダムに左右反転(p=probability)
                                    ut.RandomHorizontalFlip(p=0.5),
                                    ut.RandomVerticalFlip(p=0.5),
                                    ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    train_dataset = Covid19_Loader(root_dir=cfg.Covid19.root_dir, dataset_type='train', pseudo_path=pseudo_path, transform=train_transform)
    val_dataset = Covid19_Loader(root_dir=cfg.Covid19.root_dir, dataset_type='val', transform=val_transform)
    pseudo_dataset = Covid19_Loader(root_dir=cfg.Covid19.root_dir, dataset_type='pseudo', transform=val_transform)

    #これらをやると高速化、しかしHykwでは処理が重くなる
    #num_workers=os.cpu_count(),pin_memory=True
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.parameter.batch_size,
                                                shuffle=True,
                                                drop_last=False)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.parameter.val_size,
                                            shuffle=False,
                                            drop_last=False)

    pseudo_loader = torch.utils.data.DataLoader(pseudo_dataset,
                                            batch_size=cfg.parameter.val_size,
                                            shuffle=False,
                                            drop_last=False)

    return train_loader, val_loader, pseudo_loader


############## train関数 ##############
def train(model, train_loader, criterion, IoU_calculator, optimizer, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output, feature = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()
        ###精度の計算###
        IoU_calculator.update(output, targets, mode="softmax")
        
    IoU, mean_IoU = IoU_calculator()
    return sum_loss / (batch_idx + 1), IoU, mean_IoU


############## validation関数 ##############
def val(model, val_loader, criterion, IoU_calculator, device):
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
            output, feature = model(inputs)


            loss = criterion(output, targets)
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            IoU_calculator.update(output, targets, mode="softmax")
        
        IoU, mean_IoU = IoU_calculator()
    return sum_loss / (batch_idx + 1), IoU, mean_IoU

############## Make Pseudo関数 ##############
def make_pseudo(model, pseudo_loader, Pseudo_Maker, Pseudo_calculator, device):
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
            output, feature = model(inputs)

            pse_image = Pseudo_Maker(output)
            Pseudo_calculator.update(pse_image, targets)

    ACC = Pseudo_calculator()
    #画像番号初期化
    Pseudo_Maker.reset()

    return ACC



@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    root_dir = os.getcwd().split("/")[:-4]
    root_dir = os.path.join("/", *root_dir)

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #入力チャンネル、出力クラス設定
    in_channel = 3
    n_class = 4

    # モデル設定
    model = UNet(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    #精度計算、疑似ラベル、クラス定義
    IoU_calculator = IoU(n_class)
    Pseudo_calculator = Pseudo_accracy(n_class)
    if not os.path.exists(f"result/iteration0001"):
        os.mkdir(f"result/iteration0001")
    Pseudo_Maker = Make_Pseudo(root_dir="result/iteration0001",color=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], threshold=cfg.parameter.certainty)

    #criterion = IoULoss(n_class)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_IoU = 0.0

    #何回疑似ラベルを作成し、学習するか
    for iter_ in range(1, cfg.parameter.num_iters + 1):
        if not os.path.exists(f"result/iteration{iter_:04}"):
            os.mkdir(f"result/iteration{iter_:04}")
        
        #疑似ラベルの作成ディレクトリ更新
        Pseudo_Maker.update_rood_dir(f"result/iteration{iter_:04}")

        PATH_1 = f"result/iteration{iter_:04}/train.txt" #train
        PATH_2 = f"result/iteration{iter_:04}/validation.txt" #val
        PATH_3 = f"result/iteration{iter_:04}/pseudo.txt" #pseudo

        with open(PATH_1, mode='w') as f:
            f.write("epoch\ttrain loss\n")
        with open(PATH_2, mode='w') as f:
            f.write("epoch\tvalidation loss\tmIoU\n")
        with open(PATH_3, mode='w') as f:
            pass
        
        #1つ前のiterationファイルを確認し、ないならNone
        if os.path.exists(f"result/iteration{iter_-1:04}"):
            pseudo_path = f"result/iteration{iter_-1:04}"
        else:
            pseudo_path = None

        # データ読み込み+初期設定
        train_loader, val_loader, pseudo_loader = dataload(cfg, pseudo_path)

        #スケジューラー定義
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=cfg.scheduler.first,
                                                cycle_mult=cfg.scheduler.mult,
                                                max_lr=cfg.scheduler.max,
                                                min_lr=cfg.scheduler.min,
                                                warmup_steps=cfg.scheduler.warmup,
                                                gamma=cfg.scheduler.gamma)

        #best_IoU 初期化
        best_IoU = 0.0

        for epoch in range(cfg.parameter.epoch):
            # train
            train_loss, train_IoU, train_mIoU = train(model, train_loader, criterion, IoU_calculator, optimizer, device)
            # train
            val_loss, val_IoU, val_mIoU = val(model, val_loader, criterion, IoU_calculator, device)

            ##### 結果表示 #####
            result_text = "iter:{:3d} Epoch:{:5d}/{:5d} Train Loss:{:.5f} Train mIoU:{:.3%} val Loss:{:.5f} val mIoU:{:.3%}"
            result_text = result_text.format(iter_,
                                            epoch + 1,
                                            cfg.parameter.epoch,
                                            train_loss,
                                            train_mIoU,
                                            val_loss,
                                            val_mIoU)
            
            print(result_text)

            ##### 出力結果を書き込み #####
            with open(PATH_1, mode='a') as f:
                f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
            with open(PATH_2, mode='a') as f:
                f.write("{}\t{:.5f}\t{:.2%}\n".format(epoch+1, val_loss, val_mIoU))

            #モデル保存 & 疑似ラベル作成
            if train_mIoU > best_IoU:
                best_IoU = train_mIoU
                PATH = "result/model.pth"
                torch.save(model.state_dict(), PATH)
                pseudo_ACC = make_pseudo(model, pseudo_loader, Pseudo_Maker, Pseudo_calculator, device)

            #最初にやると一つずれるため、スケジューラーの更新は最後
            scheduler.step()
        #end for epoch

        # pseudo_ACCは必ず定義される
        with open(PATH_3, mode='a') as f:
            f.write("\t")
            for i in range(len(pseudo_ACC["Accuracy"]) - 1):
                f.write(f"class_{i}\t")
            f.write("mean\n")
            
            for k, v in pseudo_ACC.items():
                f.write(f"{k}\t")
                for item in v:
                    f.write(f"{item:.2%}\t")
                f.write("\n")

            pass


############## main ##############
if __name__ == '__main__':
    main()

