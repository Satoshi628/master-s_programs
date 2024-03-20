#coding: utf-8
#----- 標準ライブラリ -----#
import matplotlib.pyplot as plt
import hydra
import os
import argparse
import random

#----- 専用ライブラリ -----#
import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import OISTLoader, collate_delete_PAD
import utils.transforms as ut
from utils.evaluation import Object_Detection
from models.Unet_3D import UNet_3D, UNet_3D_FA

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    
    test_transform = ut.Compose([])
    
    #OIST
    if cfg.dataset == "OIST":
        test_dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/LowDensity_mov", mode="test", data="Raw_Low", length=16, transform=test_transform)
    
    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=False,
        drop_last=False)
    
    return test_loader


############## test関数 ##############
def test(model, val_loader, test_function, device):
    # モデル→推論モード
    model.eval()
    coord = []
    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets, point) in enumerate(tqdm(val_loader, leave=False)):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            point = point[:, :, :, 2:].cuda(device, non_blocking=True)
            point = point.long()

            
            # 入力画像をモデルに入力
            predicts, feature, model_coord = model(inputs)
            for p in model_coord.flatten(0, 1):
                coord.append(p)
            """
            #back_images = predicts[0, 0]
            back_images = inputs[0, 0]
            for frame,frame_image in enumerate(back_images):  #length回実行
                coordinate = test_function.coordinater(frame_image[None, None])
                frame_image = frame_image.to('cpu').detach().numpy().copy()
                frame_image = (frame_image * 255).astype(np.uint8)
                coordinate = coordinate[0].to('cpu').detach().numpy().copy()

                plt.clf()
                plt.imshow(frame_image, cmap="gray")
                for coord in coordinate:
                    plt.scatter(coord[0], coord[1], s=30, marker="x", c="r")
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.savefig(f"result/{frame:04}.png")
            #assert False
            """

            #複数回preditctsが出力されるが、一回だけ精度計算するようにする。
            if batch_idx == 0:
                #最初の処理ではlengthの半分の出力を計算する
                for frame_idx in range(predicts.shape[2] // 2):
                    coordinate = test_function.coordinater(predicts[:, :, frame_idx])  #predictsの大きさは[batch,1,T,H,W]
                    label_point = point[:, frame_idx]
                    test_function.calculate(coordinate, label_point)
            else:#中間の処理では8番目(idxでは7)の出力を計算する。
                coordinate = test_function.coordinater(predicts[:, :, 7])
                label_point = point[:, 7]
                test_function.calculate(coordinate, label_point)
        
        #最後の処理ではlengthの後半を計算する。
        for frame_idx in range(predicts.shape[2] // 2):
            coordinate = test_function.coordinater(predicts[:, :, frame_idx + 8])  #predictsの大きさは[batch,1,T,H,W]
            label_point = point[:, frame_idx + 8]
            test_function.calculate(coordinate, label_point)

        #Detectionの精度を出力
        print(len(coord))
        detection_accuracy = test_function()
        
    return detection_accuracy


@hydra.main(config_path="config", config_name='test_backbone.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    print(f"dataset     :{cfg.dataset}")

    #dataset判定
    if not cfg.dataset in ["C2C12", "OIST"]:
        raise ValueError("hydraのdatasetパラメータが違います.datasetはC2C12,OISTのみ値が許されます.")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    # モデル設定
    #model = UNet_3D(in_channels=1, n_classes=1).cuda(device)
    # モデル設定
    if cfg.parameter.assignment:
        model = UNet_3D_FA(in_channels=1, n_classes=1,
                            feature_num=cfg.parameter.feature_num,
                            overlap_range=cfg.parameter.overlap_range,
                            noise_strength=cfg.parameter.noise_strength).cuda(device)
    else:
        model = UNet_3D(in_channels=1, n_classes=1,
                        noise_strength=cfg.parameter.noise_strength).cuda(device)
    model.noise_strength = cfg.parameter.noise_strength


    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    #評価指標計算クラス定義
    test_function = Object_Detection(noise_strength=cfg.parameter.noise_strength, pool_range=11)

    # データ読み込み+初期設定
    test_loader = dataload(cfg)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    Detection = test(model, test_loader, test_function, device)
    
    PATH = "result/Detection_accuracy.txt"

    with open(PATH, mode='w') as f:
        for k in Detection.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in Detection.values():
            f.write(f"{v}\t")
        

    #Accuracy, Precition, Recall, F1_Score
    print("Detection Evalution")
    for k, v in Detection.items():
        print(f"{k}".ljust(20) + f":{v}")



############## main ##############
if __name__ == '__main__':
    main()
    
