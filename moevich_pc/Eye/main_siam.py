#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
from time import sleep

#----- External Library -----#
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import (Compose,
    Normalize,
    Resize,
    RandomCrop,
    CenterCrop,
    Pad,
    RandomHorizontalFlip,
    RandomVerticalFlip)
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm

#----- My Module -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader, Eye_Loader_traintest, Eye_Loader_ARMS2
from models.resnet2D import resnet2D50, resnet2D18, resnet2D18_atten, resnet2D18_siam, resnet2D50_siam, resnet18_pretrained
from models.vit import ViT
from utils.loss import Contrastive_Loss

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = Compose([ndarrayToTensor(),
                                RandomCrop(size=(800, 800)),  # random crop(945x945 -> 800x800)
                                RandomHorizontalFlip(p=0.5), # left right flip(p=probability)
                                RandomVerticalFlip(p=0.5), # up down flip(p=probability)
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    val_transform = Compose([ndarrayToTensor(),
                                CenterCrop(size=(800,800)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_dataset1 = Eye_Loader_ARMS2(root_dir=cfg.root_dir, dataset_type='train', split=cfg.split, transform=train_transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    train_dataset2 = Eye_Loader_ARMS2(root_dir=cfg.root_dir, dataset_type='train', split=cfg.split, transform=train_transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    val_dataset = Eye_Loader_ARMS2(root_dir=cfg.root_dir, dataset_type='val', split=cfg.split, transform=val_transform, use_age=cfg.use_age, use_sex=cfg.use_sex)
    
    
    train_loader1 = torch.utils.data.DataLoader(train_dataset1,
                                                batch_size=cfg.batch_size,
                                                shuffle=True,
                                                drop_last=True)

    train_loader2 = torch.utils.data.DataLoader(train_dataset2,
                                                batch_size=cfg.batch_size,
                                                shuffle=True,
                                                drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.val_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader1, train_loader2, val_loader


############## pre train関数 ##############
def pre_train(model, train_loader1, train_loader2, criterion, optimizer, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, ((inputs1, targets1), (inputs2, targets2)) in tqdm(enumerate(zip(train_loader1, train_loader2)), total=len(train_loader1), leave=False):
        #GPUモード高速化
        inputs1 = inputs1.cuda(device, non_blocking=True)
        targets1 = targets1.cuda(device, non_blocking=True)
        inputs2 = inputs2.cuda(device, non_blocking=True)
        targets2 = targets2.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets1 = targets1.long()

        # 入力画像をモデルに入力
        out1, out2 = model.pre_trainning(inputs1, inputs2)

        # 損失計算
        #output.size() => [batch,1] output[:,0].size() => [batch]
        loss = criterion(out1, out2, targets1, targets2)

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

############## train関数 ##############
def train(model, train_loader1, criterion, optimizer, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    total = 0
    correct = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader1), total=len(train_loader1), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output = model(inputs)

        # 損失計算
        #output.size() => [batch,1] output[:,0].size() => [batch]
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
        total += output.size(0)
        correct += (targets == output.max(dim=-1)[1]).sum()
    
    return sum_loss/(batch_idx+1), (correct/total).item()

############## val関数 ##############
def val(model, test_loader, criterion, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    total = 0
    correct = 0
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
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
            total += output.size(0)
            correct += (targets == output.max(dim=-1)[1]).sum()
    
    return sum_loss / (batch_idx + 1), (correct/total).item()


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.train_conf.epoch}")
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/val.txt"

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\ttrain acc\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tval loss\tval acc\n")
    
    in_channels = 3
    if cfg.dataset.use_age:
        in_channels += 1
    if cfg.dataset.use_sex:
        in_channels += 1
    
    # モデル設定
    if cfg.train_conf.model == "Resnet18":
        model = resnet2D18(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet18_siam":
        model = resnet2D18_siam(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet50_siam":
        model = resnet2D50_siam(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet18_atten":
        model = resnet2D18_atten(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet50":
        model = resnet2D50(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "resnet18_pre":
        model = resnet18_pretrained(num_classes=2).cuda(device)
    if cfg.train_conf.model == "ViT":
        model = ViT(image_size = 800,
            patch_size = 32,
            num_classes = 4,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            ).cuda(device)


    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    #model = apex.parallel.convert_syncbn_model(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader1, train_loader2, val_loader = dataload(cfg.dataset)

    criterion = Contrastive_Loss(margin=cfg.train_conf.margin)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    #pre trainning
    for epoch in range(cfg.train_conf.epoch):
        scheduler.step()
        loss = pre_train(model, train_loader1, train_loader2, criterion, optimizer, device)
        print(f"Epoch{epoch + 1:3d}/{cfg.train_conf.epoch:3d} contrastive Loss:{loss*1000:.5f}")

    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)
    
    model.CNN_lock()

    best_acc = 0.0
    best_loss = 999.0
    # args.num_epochs = max epoch
    for epoch in range(cfg.train_conf.epoch):
        scheduler.step()

        # train
        train_loss, train_acc = train(model, train_loader1, criterion, optimizer, device)

        # val
        val_loss, val_acc = val(model, val_loader, criterion, device)

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} Train Loss:{:.5f} train acc:{:.3%} val Loss:{:.5f} val acc:{:.3%} ".format(
            epoch + 1,
            cfg.train_conf.epoch,
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
            PATH = "result/model_val_acc.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
        if train_loss <= best_loss:
            best_loss = train_loss
            PATH = "result/model_train_loss.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)

    print("最高accurary:{:.3%}".format(best_acc))


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
    