#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep
import zipfile

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

#----- 自作モジュール -----#
from utils.dataset import  Drive_Loader_Test
import utils.utils as ut
from models.Unet import UNet
from models.Deeplabv3 import Deeplabv3


############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = ut.Compose([ut.ToTensor(),  # 0~255 ==> 0~1 + Tensor型に変換
                                    ut.DRIVE_Pad(),
                                    ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])

    test_dataset = Drive_Loader_Test(root_dir=cfg.root_dir, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            pin_memory=True)

    return test_loader

############## validation関数 ##############
def make_predict(model, val_loader, device):

    if not os.path.exists("images"):
        os.mkdir("images")
    
    image_number = 1

    # モデル→推論モード
    model.eval()
    
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)

            # 入力画像をモデルに入力
            output = model(inputs)
            
            class_idx = output.max(dim=1)[1]
            class_idx = class_idx[:, 8:, 11:] * 255
            class_img = class_idx.to("cpu").detach().clone().numpy().astype(np.uint8)

            for img in class_img:
                img = Image.fromarray(img)
                img.save(f"images/{image_number}.png")
                image_number += 1

    # zip圧縮
    with zipfile.ZipFile("test.zip", "w") as zip_f:
        for i in range(1, 21):
            zip_f.write(F"images/{i}.png",arcname=f"{i}.png")


    return


@hydra.main(config_path="config", config_name='DRIVE_test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')
    
    #データセットの設定
    model_name = cfg.dataset.model
    in_channel = cfg.dataset.input_channel
    n_class = cfg.dataset.classes

    # モデル設定
    if model_name == "U-Net":
        model = UNet(in_channels=in_channel, n_classes=n_class, channel=32).cuda(device)
    if model_name == "DeepLabv3":
        model = Deeplabv3(n_class).cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # データ読み込み+初期設定
    test_loader = dataload(cfg.dataset)
    make_predict(model, test_loader, device)


############## main ##############
if __name__ == '__main__':
    start = time.time()

    main()

    end = time.time()

    run_time = end - start
    with open("time.txt", mode='a') as f:
        f.write("{:.8f}\n".format(run_time))
    