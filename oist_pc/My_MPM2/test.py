#coding: utf-8
#----- 標準ライブラリ -----#
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
from tqdm import tqdm

#----- 自作ライブラリ -----#
from utils.dataset import C2C12Loader, OISTLoader, OISTLoader_low_test, PTCLoader
import utils.utils_C2C12 as uc
import utils.utils_OIST as uo
from utils.MPM_to_track import MPM_to_Cell, Cell_Tracker, division_process, Cell_Tracker_non_division
from utils.evaluation import Precition_Recall, Object_Tracking, Object_Detection
from U_net import UNet

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    if cfg.dataset == "C2C12":
        test_transform = uc.Compose([
                                    #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
        
        test_dataset = C2C12Loader(use_data_path=cfg.dataset.test_data,
                                    Annotater=cfg.dataset.Annotater,
                                    use_iter=[1],
                                    transform=test_transform)
    if cfg.dataset == "OIST":
        test_transform = uo.Compose([
                                    #ut.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # データ標準化
                                    ])
        
        test_dataset = OISTLoader(root_dir=cfg.OIST.root_dir,
                                    Staning=cfg.OIST.Staning,
                                    split=cfg.OIST.split,
                                    mode="test",
                                    use_iter=[1],
                                    transform=test_transform)
        #root = "/mnt/kamiya/dataset/OIST/Sim_Low_Density_mov"
        #test_dataset = OISTLoader_low_test(root_dir=root,
        #                            Staning=cfg.OIST.Staning,
        #                            split=cfg.OIST.split,
        #                            use_iter=[1],
        #                            transform=test_transform)
    if cfg.dataset == "PTC":
        ### data augmentation + preprocceing ###
        test_transform = uo.Compose([])
        
        # dataset = [image, label]
        # datatype = "train" or "val" or "test", transform = augmentation and preprocceing
        
        test_dataset = PTCLoader(root_dir=cfg.PTC.root_dir,
                                    Staning=cfg.PTC.Staning,
                                    Density=cfg.PTC.Density,
                                    split=cfg.PTC.split,
                                    mode="test",
                                    use_iter=[1],
                                    transform=test_transform)
    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    # dataloader作成高速化(大規模データだったら有効)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.parameter.batch_size, shuffle=False, drop_last=False)

    
    return test_loader


############## validation関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0
    
    predict_MPM = torch.tensor([])
    tracking_fnc = Object_Tracking()
    detection_fnc = Object_Detection()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            #targets = targets.cuda(device, non_blocking=True)

            # 入力画像をモデルに入力
            output = model(inputs)

            predict_MPM = torch.cat([predict_MPM, output.to("cpu")], dim=0)
            
        #predict_image_save(inputs, output, targets)
        cell = MPM_to_Cell(predict_MPM)
        MPM_track = Cell_Tracker(predict_MPM, cell)
        #MPM_track = Cell_Tracker_non_division(predict_MPM, cell)
        #MPM_track = division_process(MPM_track) #増えすぎだめ

        label_cell_track = test_loader.dataset.CP_data

        val = Precition_Recall()

        save_track = MPM_track[:, [0, 1, 2, 3]].to('cpu').numpy().copy()
        np.save('result/track', save_track)

        tracking_fnc.update(MPM_track[:, [0, 1, 2, 3]], label_cell_track)
        detection_fnc.update(MPM_track[:, [0, 2, 3]], label_cell_track[:, [0, 2, 3]])
        tracking_acc = tracking_fnc()
        detection_acc = detection_fnc()
    return detection_acc, tracking_acc

@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.parameter.gpu}")
    print(f"multi GPU   :{cfg.parameter.multi_gpu}")
    print(f"batch size  :{cfg.parameter.batch_size}")
    
    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test100.txt"

    with open(PATH, mode='w') as f:
        f.write("Detection Accuracy\n")
        f.write("Accuracy\tPrecision\tRecall\tF1 Score\n")

    # モデル設定
    model = UNet(n_channels=2, n_classes=3, bilinear=True).cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.parameter.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # データ読み込み+初期設定
    test_loader = dataload(cfg)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    detection, tracking = test(model, test_loader, device)
    with open(PATH, mode='a') as f:
        f.write(
            "{0[0]:.4f}\t{0[1]:.4f}\t{0[2]:.4f}\t{0[3]:.4f}\n".format(detection))
        f.write("Tracking Accuracy\n")
        for k in tracking.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in tracking.values():
            f.write(f"{v:.4f}\t")

    #Accuracy, Precition, Recall, F1_Score
    print("Detection Evalution")
    print("Accuracy     :{:.2%}".format(detection[0]))
    print("Precition    :{:.2%}".format(detection[1]))
    print("Recall       :{:.2%}".format(detection[2]))
    print("F1 Score     :{:.2%}".format(detection[3]))

    print("Tracking Evalution")
    for k, v in tracking.items():
        print(f"{k}".ljust(20) + f":{v}")
    

############## main ##############
if __name__ == '__main__':
    main()
    
