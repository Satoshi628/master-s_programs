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
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip)
import torch.nn as nn
from tqdm import tqdm

#----- My Module -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader, Eye_Loader_traintest
from models.resnet2D import resnet2D50, resnet2D18_atten
from models.vit import ViT
from models.Unet import UNet


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

    train_dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='train', split=cfg.split, transform=train_transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    val_dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='val', split=cfg.split, transform=val_transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    
    #train_dataset = Eye_Loader_traintest(root_dir=cfg.root_dir, dataset_type='train', transform=train_transform)
    #val_dataset = Eye_Loader_traintest(root_dir=cfg.root_dir, dataset_type='train', transform=val_transform)

    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=2)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.val_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader

############## train関数 ##############
def train(model, teacher, train_loader, criterion, criterion_seg, optimizer, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    total = 0
    correct = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output, atten = model(inputs)
        seg = F.interpolate(teacher(inputs), scale_factor=1/4, mode='bilinear',)

        # 損失計算
        #output.size() => [batch,1] output[:,0].size() => [batch]
        loss = criterion(output, targets)
        loss_seg = criterion_seg(atten, seg[:,1:])
        loss = loss + loss_seg*0.033

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
            output, _ = model(inputs)

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
        model = resnet2D18_atten(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet50":
        model = resnet2D50(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
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
    
    teacher = UNet(in_channels=3, n_classes=2, channel=32).cuda(device)
    teacher = torch.nn.DataParallel(teacher)
    model_path = "/mnt/hdd1/kamiya/code/Eye/teacher/hard/result/model.pth"
    teacher.load_state_dict(torch.load(model_path))
    teacher.eval()

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    #model = apex.parallel.convert_syncbn_model(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg.dataset)

    criterion = nn.CrossEntropyLoss()
    criterion_seg = nn.MSELoss()

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_acc = 0.0
    best_loss = 999.0
    # args.num_epochs = max epoch
    for epoch in range(cfg.train_conf.epoch):
        scheduler.step()

        # train
        train_loss, train_acc = train(model, teacher, train_loader, criterion, criterion_seg, optimizer, device)

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
        if train_loss < best_loss:
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
    