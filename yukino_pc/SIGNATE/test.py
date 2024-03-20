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
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.dataset import SIGNATE_TECHNOPRO_Loader
from utils.evaluation import AUROC
import torchvision.models as models


############## dataloader関数##############
def dataload(dataset_name, cfg):

    test_transform = tf.Compose([tf.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                tf.Resize(size=(64, 64)),
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                ])

    test_dataset = SIGNATE_TECHNOPRO_Loader(root_dir=cfg.root_dir, dataset_type='test', transform=test_transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False)

    return test_loader


############## test関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    result = []
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, image_names) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)

            # 入力画像をモデルに入力
            output = model(inputs)
            output = F.softmax(output, dim=-1)[:, -1]
            #二値化
            # output = (output > 0.5)*1.0
            result.extend([[image_names[idx], f"{output[idx].item():.16f}"] for idx in range(len(output))])
        
        result = np.array(result, dtype=str)
        np.savetxt("result.csv", result, delimiter=',', fmt=["%s", "%s"])


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

    # モデル設定
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))
    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.test_conf.multi_gpu else model
    # データ読み込み+初期設定
    test_loader = dataload(dataset_name, dataset_cfg)

    test(model, test_loader, device)


############## main ##############
if __name__ == '__main__':
    main()
