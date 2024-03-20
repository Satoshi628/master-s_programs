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
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import (Covid19_Loader,
    Drosophila_Loader,
    Cityscapse_Loader,
    Drive_Loader,
    ACDC_Loader,
    Synapse_Multi_Organ_Loader,
    BRATS2013_Loader,
    rTg4510_Loader)
from utils.evaluation import IoU, Dice
import utils.utils as ut
from utils.loss import IoULoss, get_class_rate_weight, get_class_balancing_weight, Contrastive_Loss_moepy, Contrastive_Loss_moepy5
from models.Unet import UNet_class_con
from models.Deeplabv3 import Deeplabv3


class Intermission():
    def __init__(self, break_time=60 * 1, works_time=60 * 60 * 2):
        super().__init__()
        self.break_T = break_time
        self.works_T = works_time
        self.start_T = time.time()

    def __call__(self):
        now_time = time.time()
        if now_time - self.start_T > self.works_T:
            print(f"{self.break_T//60}分休憩")
            time.sleep(self.break_T)
            self.start_T = time.time()

def build_transform(dataset_name):
    ### data augmentation + preprocceing ###
    if dataset_name == "Covid19":
        train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    
    if dataset_name == "Drosophila":
        train_transform = None
        val_transform = None
    
    if dataset_name == "CityScapes":
        train_transform = ut.Compose([ut.RandomCrop(size=(512, 1024)),  # ランダムクロップ(size=512x1024)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    
    if dataset_name == "DRIVE":
        train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.DRIVE_Pad(),
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    if dataset_name == "ACDC":
        train_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.RandomCrop_tensor(size=(224, 224)),  # ランダムクロップ(size=224x224)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    if dataset_name == "Synapse_Multi_Organ":
        train_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.RandomCrop_tensor(size=(224, 224)),  # ランダムクロップ(size=224x224)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ])

        val_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    if dataset_name == "BRATS2013":
        train_transform = ut.Compose([ut.Numpy2Tensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                        ut.RandomCrop_tensor(size=(224, 224)),  # ランダムクロップ(size=224x224)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5),
                                        ])

        val_transform = ut.Compose([ut.Numpy2Tensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])
    
    if dataset_name == "Mouse_brain":
        train_transform = ut.Compose([ut.RandomCrop(size=(256, 256)),  # ランダムクロップ(size=256x256)
                                        # ランダムに左右反転(p=probability)
                                        ut.RandomHorizontalFlip(p=0.5),
                                        ut.RandomVerticalFlip(p=0.5)
                                        ])

        val_transform = ut.Compose([])

    return train_transform, val_transform

############## dataloader関数##############
def dataload(dataset_name, cfg):
    ### data augmentation + preprocceing ###
    train_transform, val_transform = build_transform(dataset_name)

    if dataset_name == "Covid19":
        train_dataset = Covid19_Loader(root_dir=cfg.root_dir, dataset_type='train', transform=train_transform)
        train_dataset.few_choise(use_ratio=cfg.use_ratio)
        val_dataset = Covid19_Loader(root_dir=cfg.root_dir, dataset_type='val', transform=val_transform)
    
    if dataset_name == "Drosophila": #no few
        train_dataset = Drosophila_Loader(rootdir=cfg.root_dir, val_area=cfg.val_area, split='train', iteration_number=12 * cfg.batch_size)
        val_dataset = Drosophila_Loader(rootdir=cfg.root_dir, val_area=cfg.val_area, split='val')
    
    if dataset_name == "CityScapes":
        train_dataset = Cityscapse_Loader(root_dir=cfg.root_dir, dataset_type='train', transform=train_transform)
        val_dataset = Cityscapse_Loader(root_dir=cfg.root_dir, dataset_type='val', transform=val_transform)
    
    if dataset_name == "DRIVE":
        train_dataset = Drive_Loader(root_dir=cfg.root_dir, mode="train", split=cfg.split, transform=train_transform)
        train_dataset.few_choise(use_ratio=cfg.use_ratio)
        val_dataset = Drive_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=val_transform)
    
    if dataset_name == "ACDC":
        train_dataset = ACDC_Loader(root_dir=cfg.root_dir, mode="train", split=cfg.split, transform=train_transform)
        train_dataset.few_choise(use_ratio=cfg.use_ratio)
        val_dataset = ACDC_Loader(root_dir=cfg.root_dir, mode="val", split=cfg.split, transform=val_transform)
    
    if dataset_name == "Synapse_Multi_Organ":
        train_dataset = Synapse_Multi_Organ_Loader(root_dir=cfg.root_dir, mode="train", split=cfg.split, transform=train_transform)
        train_dataset.few_choise(use_ratio=cfg.use_ratio)
        val_dataset = Synapse_Multi_Organ_Loader(root_dir=cfg.root_dir, mode="val", split=cfg.split, transform=val_transform)
    
    if dataset_name == "BRATS2013":
        train_dataset = BRATS2013_Loader(root_dir=cfg.root_dir, mode="train", split=cfg.split, transform=train_transform)
        train_dataset.few_choise(use_ratio=cfg.use_ratio)
        val_dataset = BRATS2013_Loader(root_dir=cfg.root_dir, mode="val", split=cfg.split, transform=val_transform)

    if dataset_name == "Mouse_brain":
        train_dataset = rTg4510_Loader(root_dir=cfg.root_dir, mode="train", split=cfg.split, transform=train_transform)
        val_dataset = rTg4510_Loader(root_dir=cfg.root_dir, mode="val", split=cfg.split, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=2,
                                                pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.val_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            pin_memory=True)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader

############## train関数 ##############
def train(cfg, model, train_loader, criterion, optimizer, eval_calculater, device):
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
        output, feature = model(inputs, train=True)

        # 損失計算
        if cfg.train_conf.use_Loss == "IoU":
            loss = criterion(output, targets)
        elif cfg.train_conf.use_Loss == "CE":
            loss = criterion(output, targets)
        elif cfg.train_conf.use_Loss == "CL-moepy":
            loss = criterion(output, feature, targets)


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
        eval_calculater.update(output, targets)
    eval_result = eval_calculater()
    
    return sum_loss/(batch_idx+1), eval_result

############## validation関数 ##############
def val(cfg, model, val_loader, criterion, eval_calculater, device):
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
            output, feature = model(inputs, train=True)

            # 損失計算
            if cfg.train_conf.use_Loss == "IoU":
                loss = criterion(output, targets)
            elif cfg.train_conf.use_Loss == "CE":
                loss = criterion(output, targets)
            elif cfg.train_conf.use_Loss == "CL-moepy":
                loss = criterion(output, feature, targets)
            
            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###
            eval_calculater.update(output, targets)

        eval_result = eval_calculater()
    return sum_loss / (batch_idx + 1), eval_result


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.train_conf.epoch}")
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")
    print(f"dataset     :{cfg.dataset.name}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    #データセットの設定
    dataset_name = cfg.dataset.name
    dataset_cfg = eval(f"cfg.dataset.{dataset_name}")
    model_name = dataset_cfg.model
    in_channel = dataset_cfg.input_channel
    n_class = dataset_cfg.classes

    #精度計算クラス定義
    if cfg.train_conf.use_Eval == "IoU":
        eval_calculater = IoU(n_class)
    if cfg.train_conf.use_Eval == "Dice":
        eval_calculater = Dice(n_class)
    
    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"
    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\t")
        for k in eval_calculater.keys():
            f.write(f"{k}\t")
        f.write("\n")
    with open(PATH_2, mode='w') as f:
        f.write("epoch\ttrain loss\t")
        for k in eval_calculater.keys():
            f.write(f"{k}\t")
        f.write("\n")
    
    # モデル設定
    if model_name == "U-Net":
        model = UNet_class_con(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "DeepLabv3":
        model = Deeplabv3(n_class).cuda(device)

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    #model = apex.parallel.convert_syncbn_model(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, val_loader = dataload(dataset_name, dataset_cfg)

    class_count = train_loader.dataset.get_class_count()
    if cfg.train_conf.use_weight == "Blance":
        weight = get_class_balancing_weight(class_count, n_class)
        weight = weight.to(device).float()
    if cfg.train_conf.use_weight == "Rate":
        weight = get_class_rate_weight(class_count, n_class)
        weight = weight.to(device).float()
    if cfg.train_conf.use_weight == "None":
        weight = None

    #criterion = nn.CrossEntropyLoss(ignore_index=19)
    if cfg.train_conf.use_Loss == "IoU":
        criterion = IoULoss(n_class, weight=weight)
    elif cfg.train_conf.use_Loss == "CE":
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif cfg.train_conf.use_Loss == "CL-moepy":
        criterion = Contrastive_Loss_moepy5(n_class,
        choice_type=cfg.train_conf.choice_type,
        edge_range=cfg.train_conf.edge_range,
        n=cfg.train_conf.choice_num,
        weight=weight,
        loss_type=cfg.train_conf.loss_type,
        k=cfg.train_conf.kappa)
        print("tvmf ok")
    else:
        raise KeyError("使用するLossが正しくありません")

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD
    # Initialization

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)
    

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    Break_Time = Intermission()
    # valを実行しないときがあるため
    val_loss, val_mIoU = 0.0, {k: 0.0 for k in eval_calculater.keys()}
    best_IoU = 0.0
    # args.num_epochs = max epoch
    for epoch in range(cfg.train_conf.epoch):
        # train
        train_loss, train_result = train(cfg, model, train_loader, criterion, optimizer, eval_calculater, device)

        # validation
        if epoch >= dataset_cfg.val_start * cfg.train_conf.epoch:
            val_loss, val_result = val(cfg, model, val_loader, criterion, eval_calculater, device)

        ##### 結果表示 #####
        mean_eval = list(eval_calculater.keys())[-1]
        result_text = f"Epoch{epoch + 1:3d}/{cfg.train_conf.epoch:3d} \
                    Train Loss:{train_loss:.5f} train mIoU:{train_result[mean_eval]:.3%} \
                    val Loss:{val_loss:.5f} val mIoU:{val_result[mean_eval]:.3%}"
        print(result_text)

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t")
            for v in train_result.values():
                f.write(f"{v}\t")
            f.write("\n")
        
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch + 1}\t{val_loss:.4f}\t")
            for v in val_result.values():
                f.write(f"{v}\t")
            f.write("\n")

        if val_result[mean_eval] > best_IoU:
            best_IoU = val_result[mean_eval]
            PATH = "result/model.pth"
            if cfg.train_conf.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
        
        #if (epoch + 1) % 300 == 0:
        #    PATH = f"result/model_epoch{epoch+1}.pth"
        #    if cfg.train_conf.multi_gpu:
        #        torch.save(model.module.state_dict(), PATH)
        #    else:
        #        torch.save(model.state_dict(), PATH)
        scheduler.step()
        Break_Time()

    print("最高IoU:{:.3%}".format(best_IoU))


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
