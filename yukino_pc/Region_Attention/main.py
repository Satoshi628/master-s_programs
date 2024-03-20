#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Covid19_Loader, Drosophila_Loader
from utils.evaluation import IoU
import utils.utils as ut
from utils.loss import FocalLoss, IoULoss, Edge_IoULoss
from models.Unet import UNet


############## dataloader関数##############
def dataload(cfg):
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

    if cfg.dataset == "Covid19":
        train_dataset = Covid19_Loader(root_dir=cfg.Covid19.root_dir, dataset_type='train', transform=train_transform)
        val_dataset = Covid19_Loader(root_dir=cfg.Covid19.root_dir, dataset_type='val', transform=val_transform)
    
    if cfg.dataset == "Drosophila":
        train_dataset = Drosophila_Loader(rootdir=cfg.Drosophila.root_dir, val_area=cfg.Drosophila.val_area, split='train', iteration_number=12 * cfg.parameter.batch_size)
        val_dataset = Drosophila_Loader(rootdir=cfg.Drosophila.root_dir, val_area=cfg.Drosophila.val_area, split='val')
        

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.parameter.batch_size,
                                                shuffle=True,
                                                drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.parameter.val_size,
                                            shuffle=False,
                                            drop_last=False)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader


############## train関数 ##############
def train(model, train_loader, criterion, optimizer, IoU_calculator, device):
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
        IoU_calculator.update(output, targets, mode="softmax")
    IoU, mean_IoU = IoU_calculator()
    
    return sum_loss/(batch_idx+1), IoU, mean_IoU


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
            output = model(inputs)

            loss = criterion(output, targets)
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            IoU_calculator.update(output, targets, mode="softmax")

        IoU, mean_IoU = IoU_calculator()
    return sum_loss / (batch_idx + 1), IoU, mean_IoU


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.parameter.epoch}")
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")
    
    #dataset判定
    dataset = ["Covid19", "Drosophila"]
    if not cfg.dataset in ["Covid19", "Drosophila"]:
        raise ValueError("hydraのdatasetパラメータが違います.datasetは{}のみが許されます.".format(dataset))
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\tmIoU\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tvalidation loss\tmIoU\n")
    
    #データセットの入力チャンネルとクラス数を選択
    try:
        in_channel = eval(f'cfg.{cfg.dataset}.input_channel')
        n_class = eval(f'cfg.{cfg.dataset}.classes')
    except:
        if cfg.dataset == "Covid19":
            in_channel = cfg.Covid19.input_channel
            n_class = cfg.Covid19.classes
        if cfg.dataset == "Drosophila":
            in_channel = cfg.Drosophila.input_channel
            n_class = cfg.Drosophila.classes
    
    # モデル設定
    model = UNet(in_channels=in_channel, n_classes=n_class, **cfg.param).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model
    # batch size を変えたくない場合
    #model = apex.parallel.convert_syncbn_model(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #精度計算クラス定義
    IoU_calculator = IoU(n_class)

    # データ読み込み+初期設定
    train_loader, val_loader = dataload(cfg)

    #criterion = IoULoss(n_class)
    criterion = Edge_IoULoss(n_class,
                            edge_range=cfg.parameter.edge_range,
                            lamda=cfg.parameter.alpha)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)
    

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    best_IoU = 0.0
    # args.num_epochs = max epoch
    for epoch in range(cfg.parameter.epoch):

        # train
        train_loss, train_IoU, train_mIoU = train(model, train_loader, criterion, optimizer, IoU_calculator, device)
        # validation
        val_loss, val_IoU, val_mIoU  = val(model, val_loader, criterion, IoU_calculator, device)

        ##### 結果表示 #####
        result_text = "Epoch{:3d}/{:3d} Train Loss:{:.5f} train mIoU:{:.3%} val Loss:{:.5f} val mIoU:{:.3%} ".format(
            epoch + 1,
            cfg.parameter.epoch,
            train_loss,
            train_mIoU,
            val_loss,
            val_mIoU)

        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.3%}\n".format(epoch+1, train_loss, train_mIoU))
        with open(PATH_2, mode='a') as f:
            f.write("{}\t{:.4f}\t{:.3%}\n".format(epoch+1, val_loss, val_mIoU))

        if val_mIoU > best_IoU:
            best_IoU = val_mIoU
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

        scheduler.step()

    print("最高IoU:{:.3%}".format(best_IoU))


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
