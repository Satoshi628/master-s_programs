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
from utils.dataset import Eye_Loader_2classification2, Eye_Loader_2classification2_traintest
from models.resnet2D import resnet2D2_18, resnet2D2_50, ResNet2D_2
from models.vit import ViT2class


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

    train_dataset = Eye_Loader_2classification2(root_dir=cfg.root_dir, dataset_type='train', split=cfg.split, transform=train_transform, use_age=cfg.use_age, use_sex=cfg.use_sex)
    val_dataset = Eye_Loader_2classification2(root_dir=cfg.root_dir, dataset_type='val', split=cfg.split, transform=val_transform, use_age=cfg.use_age, use_sex=cfg.use_sex)
    #train_dataset = Eye_Loader_2classification2_traintest(root_dir=cfg.root_dir, dataset_type='train', transform=train_transform)
    #val_dataset = Eye_Loader_2classification2_traintest(root_dir=cfg.root_dir, dataset_type='train', transform=val_transform)
    
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
def train(model, train_loader, criterion, optimizer, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    total = 0
    correct1 = 0
    correct2 = 0
    correct_all = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output1, output2 = model(inputs)


        # 損失計算
        #output.size() => [batch,1] output[:,0].size() => [batch]
        loss1 = criterion(output1, targets[:,0])
        loss2 = criterion(output2, targets[:,1])
        loss = loss1 + loss2

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
        total += output1.size(0)
        correct1 += (targets[:,0] == output1.max(dim=-1)[1]).sum()
        correct2 += (targets[:,1] == output2.max(dim=-1)[1]).sum()
        correct_all += ((targets[:,0] == output1.max(dim=-1)[1]) & (targets[:,1] == output2.max(dim=-1)[1])).sum()
    
    return sum_loss/(batch_idx+1), (correct1/total).item(), (correct2/total).item(), (correct_all/total).item()

############## val関数 ##############
def val(model, test_loader, criterion, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    total = 0
    correct1 = 0
    correct2 = 0
    correct_all = 0
    
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
            output1, output2 = model(inputs)

            loss1 = criterion(output1, targets[:,0])
            loss2 = criterion(output2, targets[:,1])
            loss = loss1 + loss2
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            total += output1.size(0)
            correct1 += (targets[:,0] == output1.max(dim=-1)[1]).sum()
            correct2 += (targets[:,1] == output2.max(dim=-1)[1]).sum()
            correct_all += ((targets[:,0] == output1.max(dim=-1)[1]) & (targets[:,1] == output2.max(dim=-1)[1])).sum()

    
    return sum_loss/(batch_idx+1), (correct1/total).item(), (correct2/total).item(), (correct_all/total).item()



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
        f.write("epoch\ttrain loss\ttrain ARMS2 acc\ttrain CFH acc\ttrain all acc\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tval loss\tval ARMS2 acc\tval CFH acc\tval all acc\n")
    
    in_channels = 3
    if cfg.dataset.use_age:
        in_channels += 1
    if cfg.dataset.use_sex:
        in_channels += 1
    ResNet2D_2
    # モデル設定
    if cfg.train_conf.model == "Resnet18":
        model = resnet2D2_18(in_c=in_channels, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet18_2":
        model = ResNet2D_2(in_c=in_channels, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet50":
        model = resnet2D2_50(in_c=in_channels, non_local=False).cuda(device)
    if cfg.train_conf.model == "ViT":
        model = ViT2class(image_size = 800,
            patch_size = 32,
            num_classes = 2,
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
    train_loader, val_loader = dataload(cfg.dataset)

    criterion = nn.CrossEntropyLoss()

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_ARMS2_acc = 0.0
    best_CFH_acc = 0.0
    # args.num_epochs = max epoch
    for epoch in range(cfg.train_conf.epoch):
        scheduler.step()

        # train
        #sum_loss/(batch_idx+1), (correct1/total1).item(), (correct2/total2).item(), ((correct1 + correct2)/(total1 + total2)).item()
        train_loss, train_ARMS2_acc, train_CFH_acc, train_all_acc = train(model, train_loader, criterion, optimizer, device)

        # val
        val_loss, val_ARMS2_acc, val_CFH_acc, val_all_acc = val(model, val_loader, criterion, device)

        ##### 結果表示 #####
        result_text = f"Epoch{epoch + 1:3d}/{cfg.train_conf.epoch:3d} \
            Train Loss:{train_loss:.5f} \
            Train ARMS2 acc:{train_ARMS2_acc:.3%} \
            Train CFH acc:{train_CFH_acc:.3%} \
            Train ALL acc:{train_all_acc:.3%} \
            val ARMS2 acc:{val_ARMS2_acc:.3%} \
            val CFH acc:{val_CFH_acc:.3%} \
            val ALL acc:{val_all_acc:.3%}"

        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{train_ARMS2_acc:.3%}\t{train_CFH_acc:.3%}\t{train_all_acc:.3%}\n")
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch+1}\t{val_loss:.4f}\t{val_ARMS2_acc:.3%}\t{val_CFH_acc:.3%}\t{val_all_acc:.3%}\n")

        if val_ARMS2_acc > best_ARMS2_acc:
            best_ARMS2_acc = val_ARMS2_acc
            PATH = "result/model_ARMS2.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
        if val_CFH_acc > best_CFH_acc:
            best_CFH_acc = val_CFH_acc
            PATH = "result/model_CFH.pth"
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
    