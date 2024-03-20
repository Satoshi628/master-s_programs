#coding: utf-8
#----- 標準ライブラリ -----#
import matplotlib.pyplot as plt
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.dataset import Covid19_Loader, Drosophila_Loader, Cityscapse_Loader, Drive_Loader, ACDC_Loader, Synapse_Multi_Organ_Loader, BRATS2013_Loader
from utils.visualization import Seg_image
from utils.evaluation import IoU, Dice, HD
import utils.utils as ut
from models.Unet import UNet, UNet_class_con
from models.Deeplabv3 import Deeplabv3

############## dataloader関数##############


def build_transform(dataset_name):
    ### data augmentation + preprocceing ###
    if dataset_name == "Covid19":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
    
    if dataset_name == "Drosophila":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])
    
    if dataset_name == "CityScapes":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    if dataset_name == "ACDC":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    if dataset_name == "Synapse_Multi_Organ":
        test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])
    
    if dataset_name == "BRATS2013":
        test_transform = ut.Compose([ut.Numpy2Tensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ])

    return test_transform



def dataload(dataset_name, cfg):
    ### data augmentation + preprocceing ###
    test_transform = build_transform(dataset_name)

    if dataset_name == "Covid19":
        test_dataset =Covid19_Loader(root_dir=cfg.root_dir, dataset_type='test', transform=test_transform)

    if dataset_name == "Drosophila":
        test_dataset = Drosophila_Loader(rootdir=cfg.root_dir, val_area=cfg.val_area, split='test')

    if dataset_name == "CityScapes":
        test_dataset = Cityscapse_Loader(root_dir=cfg.root_dir, dataset_type='val', transform=test_transform)

    if dataset_name == "ACDC":
        test_dataset = ACDC_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=test_transform)

    if dataset_name == "Synapse_Multi_Organ":
        test_dataset = Synapse_Multi_Organ_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=test_transform)

    if dataset_name == "BRATS2013":
        test_dataset = BRATS2013_Loader(root_dir=cfg.root_dir, mode="test", split=cfg.split, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False)
    
    return test_loader


############## test関数 ##############
def test(model, test_loader, Eval_calc_list, cfg, device):
    # モデル→推論モード
    model.eval()

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

            ###精度の計算###
            for eval_calc in Eval_calc_list:
                eval_calc.update(output, targets)

        eval_result ={}
        for eval_calc in Eval_calc_list: eval_result.update(eval_calc())
        

    return eval_result


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.test_conf.gpu}")
    print(f"multi GPU   :{cfg.test_conf.multi_gpu}")
    print(f"dataset     :{cfg.dataset.name}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.test_conf.gpu) if torch.cuda.is_available() else 'cpu')


    #データセットの設定
    dataset_name = cfg.dataset.name
    dataset_cfg = eval(f"cfg.dataset.{dataset_name}")
    model_name = dataset_cfg.model
    in_channel = dataset_cfg.input_channel
    n_class = dataset_cfg.classes

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test.txt"
    with open(PATH, mode='w') as f:
        f.write("")


    # モデル設定
    if model_name == "U-Net":
        model = UNet(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "U-Net-moepy":
        model = UNet_class_con(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "DeepLabv3":
        model = Deeplabv3(n_class).cuda(device)


    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))
    #マルチGPU
    #model = torch.nn.DataParallel(model) if cfg.test_conf.multi_gpu else model


    #精度計算クラス定義
    Eval_calc_list = []
    if "IoU" in cfg.test_conf.use_Eval:
        Eval_calc_list.append(IoU(n_class))
    if "Dice" in cfg.test_conf.use_Eval:
        Eval_calc_list.append(Dice(n_class))
    if "HD" in cfg.test_conf.use_Eval:
        Eval_calc_list.append(HD(n_class))


    # データ読み込み+初期設定
    test_loader = dataload(dataset_name, dataset_cfg)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    for epoch in range(47):
        e = 300 * (epoch + 1)
        model_path = f"result/model_epoch{e}.pth"
        model.load_state_dict(torch.load(model_path))
        eval_result = test(model, test_loader, Eval_calc_list, cfg, device)

        with open(PATH, mode='a') as f:
            for v in eval_result.values():
                f.write(f"{v}\t")
            f.write("\n")

        
        #Accuracy表示
        for k, v in eval_result.items():
            print(f"{k}".ljust(20) + f"{v}")


############## main ##############
if __name__ == '__main__':
    main()
