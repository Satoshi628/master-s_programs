#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as tf

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import tanacho_DataLoader
from utils.evaluation import mAP
from utils.loss import Contrastive_Loss, Distance_Loss
from models.resnet import ResNet, ResNet_avgpool4

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([
                                    tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    val_transform = tf.Compose([tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    train_dataset = tanacho_DataLoader(transform=train_transform)
    val_dataset = tanacho_DataLoader(transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.train_conf.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=2,
                                                pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.train_conf.val_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            pin_memory=True)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader

############## train関数 ##############
def train(model, train_loader, criterion, optimizer, device):
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
        vector = model(inputs)

        # 損失計算
        loss = criterion(vector, targets)
        
        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬
        loss.backward()

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()

    return sum_loss / (batch_idx + 1)

############## validation関数 ##############
def val(model, val_loader, criterion, device):
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
            vector = model(inputs)

            # 損失計算
            loss = criterion(vector, targets)

            
            # loss溜め
            sum_loss += loss.item()

    return sum_loss / (batch_idx + 1)


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.train_conf.epoch}")
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    #データセットの設定
    in_channel = 3

    #精度計算クラス定義
    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"
    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain_loss\ttrain_mAP\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tval_loss\tval_mAP\n")
    
    # モデル設定
    #model = ResNet(resnet_type=cfg.train_conf.resnet_type, pretrained=True).cuda(device)
    model = ResNet_avgpool4(resnet_type=cfg.train_conf.resnet_type, pretrained=True).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg)

    #criterion = nn.CrossEntropyLoss()
    criterion = Contrastive_Loss()
    #criterion = Distance_Loss()

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD
    # Initialization

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)
    

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    # valを実行しないときがあるため
    best_loss = 0.0
    # args.num_epochs = max epoch
    for epoch in range(cfg.train_conf.epoch):
        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)

        # validation
        val_loss = val(model, val_loader, criterion, device)

        ##### 結果表示 #####
        result_text = f"Epoch{epoch + 1:3d}/{cfg.train_conf.epoch:3d} \
                    Train Loss:{train_loss:.5f} \
                    val Loss:{val_loss:.5f}"
        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\n")
        
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch + 1}\t{val_loss:.4f}\n")

        if val_loss > best_loss:
            best_loss = val_loss
            PATH = "result/model.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
        
        scheduler.step()

    print("最高Loss:{:.3%}".format(best_loss))


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
