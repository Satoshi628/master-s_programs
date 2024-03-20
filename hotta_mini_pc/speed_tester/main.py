#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
#----- 専用ライブラリ -----#
from tqdm import tqdm
import hydra
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import (
    Compose,
    RandomCrop,
    Resize,
    RandomResizedCrop,
    RandomHorizontalFlip,   # flip left right
    RandomVerticalFlip,  # flip up down
    # Normalize param is mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    Normalize
    )
import apex
from apex import amp, optimizers

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import TinyImageNet_Loader
from utils.evaluation import Classification
from utils.transforms import ToTensor
from models import Resnet_Classification

train_time = 0.0
val_time = 0.0

############## dataloader関数##############
def dataload(dataset_name, cfg):
    ### data augmentation + preprocceing ###
    train_transform = Compose([ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                #Resize(224),
                                RandomResizedCrop(224),
                                #RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                RandomHorizontalFlip(p=0.5),  # ランダムに左右反転(p=probability)
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    val_transform = Compose([ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                            #Resize(224),
                            RandomResizedCrop(224),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                            ])

    train_dataset = TinyImageNet_Loader(root_dir=cfg.root_dir, dataset_type='train', transform=train_transform)
    val_dataset = TinyImageNet_Loader(root_dir=cfg.root_dir, dataset_type='val', transform=val_transform)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.val_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=os.cpu_count(),
                                            pin_memory=True)

    return train_loader, val_loader


############## train関数 ##############
def train(model, train_loader, criterion, optimizer, eval_fnc, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    global train_time

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        start = time.time()
        output = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()
        
        # 誤差逆伝搬
        loss.backward()
        end = time.time()
        train_time += end - start

        # loss溜め
        sum_loss += loss.item()

        del loss  # 誤差逆伝播を実行後、計算グラフを削除

        # パラメーター更新
        optimizer.step()

        ###精度の計算###
        eval_fnc.update(output, targets)
    Top1_acc, Top5_acc = eval_fnc()
    
    return sum_loss/(batch_idx+1), Top1_acc, Top5_acc


############## validation関数 ##############
def val(model, val_loader, criterion, eval_fnc, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    global val_time
    
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
            start = time.time()
            output = model(inputs)

            loss = criterion(output, targets)
            end = time.time()
            val_time += end - start
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            eval_fnc.update(output, targets)

        Top1_acc, Top5_acc = eval_fnc()
    return sum_loss / (batch_idx + 1), Top1_acc, Top5_acc


@hydra.main(config_name='main.yaml')
def main(cfg):
    #データセットの設定
    dataset_name = cfg.dataset.name
    dataset_cfg = eval(f"cfg.dataset.{dataset_name}")
    in_channel = dataset_cfg.input_channel
    n_class = dataset_cfg.classes

    print(f"epoch               :{dataset_cfg.epoch}")
    print(f"gpu                 :{cfg.train_conf.gpu}")
    print(f"multi GPU           :{cfg.train_conf.multi_gpu}")
    print(f"train batch size    :{dataset_cfg.batch_size}")
    print(f"val batch size      :{dataset_cfg.val_size}")
    print(f"start val epoch     :{dataset_cfg.val_start*dataset_cfg.epoch}")
    print(f"dataset             :{cfg.dataset.name}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.csv"
    PATH_2 = "result/validation.csv"

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\tTop1\tTon5\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tvalidation loss\tTop1\tTon5\n")
    
    
    # モデル設定
    model = Resnet_Classification(name="resnet50", in_channels=in_channel, num_classes=n_class).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    # batch size を変えたくない場合
    #model = apex.parallel.convert_syncbn_model(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #精度計算クラス定義
    eval_fnc = Classification(n_class)

    # データ読み込み+初期設定
    train_loader, val_loader = dataload(dataset_name, dataset_cfg)

    criterion = nn.CrossEntropyLoss()

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=dataset_cfg.epoch,
                                              cycle_mult=1.,
                                              max_lr=0.001,
                                              min_lr=0.,
                                              warmup_steps=0,
                                              gamma=1.,)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    val_loss, val_top1, val_top5 = 0., 0., 0.
    best_acc = 0.0
    # args.num_epochs = max epoch
    
    for epoch in range(dataset_cfg.epoch):
        # train
        train_loss, train_top1, train_top5 = train(model, train_loader, criterion, optimizer, eval_fnc, device)
        
        # validation
        if epoch >= dataset_cfg.val_start * dataset_cfg.epoch:
            val_loss, val_top1, val_top5 = val(model, val_loader, criterion, eval_fnc, device)

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} Train Loss:{:.5f} train top1:{:.3%} train top5:{:.3%} val Loss:{:.5f} val top1:{:.3%} val top5:{:.3%}".format(
            epoch + 1,
            dataset_cfg.epoch,
            train_loss,
            train_top1,
            train_top5,
            val_loss,
            val_top1,
            val_top5)

        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.3%}\t{:.3%}\n".format(epoch + 1, train_loss, train_top1, train_top5))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.3%}\t{:.3%}\n".format(epoch + 1, val_loss, val_top1, val_top5))

        if val_top1 > best_acc:
            best_acc = val_top1
            PATH = "result/model.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)

        scheduler.step()
    
    print("最高精度:{:.3%}".format(best_acc))
    with open(PATH_2, mode='a') as f:
        f.write(f"\nbest Top1 accuracy\t{best_acc:.3%}")




############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='w') as f:
        f.write(f"run_time:{run_time:.2f} sec,train_time:{train_time:.2f} sec,val_time:{val_time:.2f} sec\n")
        f.write(f"run_time:{run_time/60:.2f} min,train_time:{train_time/60:.2f} min,val_time:{val_time/60:.2f} min\n")
