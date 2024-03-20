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
import torchvision.transforms as tf
#----- 自作モジュール -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import SIGNATE_TECHNOPRO_Loader
from utils.evaluation import AUROC
from utils.loss import get_class_rate_weight, get_class_balancing_weight
import torchvision.models as models


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

############## dataloader関数##############
def dataload(dataset_name, cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.Resize(size=(64, 64)),  # ランダムクロップ(size=256x256)
                                    # ランダムに左右反転(p=probability)
                                    tf.RandomHorizontalFlip(p=0.5),
                                    # tf.GaussianBlur((13,13), sigma=(0.1, 2.0)),
                                    tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    val_transform = tf.Compose([tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    train_dataset = SIGNATE_TECHNOPRO_Loader(root_dir=cfg.root_dir, dataset_type='train', transform=train_transform)
    val_dataset = SIGNATE_TECHNOPRO_Loader(root_dir=cfg.root_dir, dataset_type='train', transform=val_transform)
    
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
def train(model, train_loader, criterion, optimizer, eval_calculater, device):
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
        eval_calculater.update(output, targets)
    eval_result = eval_calculater()
    
    return sum_loss/(batch_idx+1), eval_result

############## validation関数 ##############
def val(model, val_loader, criterion, eval_calculater, device):
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
            eval_calculater.update(output, targets)

        eval_result = eval_calculater()
    return sum_loss / (batch_idx + 1), eval_result


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print(f"epoch       :{cfg.config.epoch}")
    print(f"gpu         :{cfg.config.gpu}")
    print(f"multi GPU   :{cfg.config.multi_gpu}")
    print(f"dataset     :{cfg.dataset.name}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.config.gpu) if torch.cuda.is_available() else 'cpu')

    #データセットの設定
    dataset_name = cfg.dataset.name
    dataset_cfg = eval(f"cfg.dataset.{dataset_name}")
    model_name = dataset_cfg.model
    in_channel = dataset_cfg.input_channel
    n_class = dataset_cfg.classes

    #精度計算クラス定義
    eval_calculater = AUROC()
    
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
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.cuda(device)


    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.config.multi_gpu else model
    #model = apex.parallel.convert_syncbn_model(model) if cfg.config.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, val_loader = dataload(dataset_name, dataset_cfg)

    class_count = train_loader.dataset.get_class_count()
    if cfg.config.use_weight == "Blance":
        weight = get_class_balancing_weight(class_count, n_class)
        weight = weight.to(device).float()
    if cfg.config.use_weight == "Rate":
        weight = get_class_rate_weight(class_count, n_class)
        weight = weight.to(device).float()
    if cfg.config.use_weight == "None":
        weight = None

    #criterion = nn.CrossEntropyLoss(ignore_index=19)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max_lr)  # Adam
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD
    # Initialization

    scheduler = CosineAnnealingWarmupRestarts(optimizer, **cfg.scheduler)
    

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    Break_Time = Intermission()
    # valを実行しないときがあるため
    best_ROC = 0.0
    epoch_result = []
    # args.num_epochs = max epoch
    for epoch in range(cfg.config.epoch):
        # train
        train_loss, train_result = train(model, train_loader, criterion, optimizer, eval_calculater, device)
        epoch_result.append(train_result)

        # validation
        val_loss, val_result = val(model, val_loader, criterion, eval_calculater, device)

        ##### 結果表示 #####
        mean_eval = list(eval_calculater.keys())[-1]
        result_text = f"Epoch{epoch + 1:3d}/{cfg.config.epoch:3d} \
                    Train Loss:{train_loss:.5f} train AUROC:{train_result[mean_eval]:.3%} \
                    val Loss:{val_loss:.5f} val AUROC:{val_result[mean_eval]:.3%}"
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

        if val_result[mean_eval] > best_ROC:
            best_ROC = val_result[mean_eval]
            PATH = "result/model.pth"
            if cfg.config.multi_gpu:
                torch.save(model.module.state_dict(), PATH)
            else:
                torch.save(model.state_dict(), PATH)
        
        scheduler.step()
        Break_Time()

    print("最高AUROC:{:.3%}".format(best_ROC))


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
