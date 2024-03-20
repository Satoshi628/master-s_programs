#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
import argparse
#----- 専用ライブラリ -----#
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import (
    Compose,
    RandomCrop,
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
from utils.utils import ToTensor
from models.resnet import resnet18, resnet34, resnet50

root_dir = "/mnt/kamiya/dataset/tiny-imagenet-200"

############## dataloader関数##############
def dataload():
    ### data augmentation + preprocceing ###
    train_transform = Compose([ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                #RandomCrop(size=(256, 256)),  # ランダムクロップ(size=320×320)
                                RandomHorizontalFlip(p=0.5),  # ランダムに左右反転(p=probability)
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    val_transform = Compose([ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                            ])

    train_dataset = TinyImageNet_Loader(root_dir=root_dir, dataset_type='train', transform=train_transform)
    val_dataset = TinyImageNet_Loader(root_dir=root_dir, dataset_type='val', transform=val_transform)

    sampler_train = DistributedSampler(train_dataset,
                                    num_replicas=world_size,
                                    rank=rank,
                                    shuffle=True,
                                    seed=0)
    sampler_val = DistributedSampler(val_dataset,
                                    num_replicas=world_size,
                                    rank=rank,
                                    shuffle=False,
                                    seed=0)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=2,
                                                sampler=sampler_train,
                                                pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            sampler=sampler_val,
                                            pin_memory=True)

    return train_loader, val_loader


############## train関数 ##############
def train(model, train_loader, criterion, optimizer, eval_fnc, device):
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
        output = model(inputs)

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
        eval_fnc.update(output, targets)
    acc = eval_fnc()
    
    return sum_loss/(batch_idx+1), acc


############## validation関数 ##############
def val(model, val_loader, criterion, eval_fnc, device):
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
            output = model(inputs)

            loss = criterion(output, targets)
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            eval_fnc.update(output, targets)

        acc = eval_fnc()
    return sum_loss / (batch_idx + 1), acc


def main():
    ##### GPU設定 #####
    device = torch.device(f'cuda:{rank}')
    
    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.csv"
    PATH_2 = "result/validation.csv"

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\tacc\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tvalidation loss\tacc\n")
    
    in_channel = 3
    n_class = 200 
    # モデル設定
    model = resnet50(in_channels=in_channel, num_classes=n_class).cuda(device)

    #マルチGPU
    #model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    model = DDP(model, device_ids=[rank])

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
    train_loader, val_loader = dataload()

    criterion = nn.CrossEntropyLoss()

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=100,
                                              cycle_mult=1.,
                                              max_lr=0.001,
                                              min_lr=0.,
                                              warmup_steps=0,
                                              gamma=1.,)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    print("len(train_loader)", len(train_loader))
    print("len(val_loader)", len(val_loader))


    val_loss, val_acc = 0., 0.
    best_acc = 0.0
    # args.num_epochs = max epoch
    for epoch in range(100):
        # train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, eval_fnc, device)
        
        # validation
        
        val_loss, val_acc = val(model, val_loader, criterion, eval_fnc, device)

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} Train Loss:{:.5f} train acc:{:.3%} val Loss:{:.5f} val acc:{:.3%} ".format(
            epoch + 1,
            100,
            train_loss,
            train_acc,
            val_loss,
            val_acc)

        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.3%}\n".format(epoch+1, train_loss, train_acc))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.3%}\n".format(epoch+1, val_loss, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            PATH = "result/model.pth"
            torch.save(model.module.state_dict(), PATH)

        scheduler.step()

    print("最高精度:{:.3%}".format(best_acc))
    dist.destroy_process_group()
    return 100


############## main ##############
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    rank = args.local_rank

    batch_size = 256

    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    

    start = time.time()

    epoch = main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write(f"{epoch}\t{run_time:.2f}\t{run_time/epoch:.3f}sec/epoch\n")
