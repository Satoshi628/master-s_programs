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
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip)
import apex
from apex import amp, optimizers
import torch.nn as nn
from tqdm import tqdm

#----- My Module -----#
from utils.utils import CT_ndarrayToTensor, Multi_CT_ndarrayToTensor
from utils.dataset import PVT_Loader_plot
from utils.visualization import Mr_sota_haruki_Grad_CAM_3D, kamiya_Grad_CAM_3D,CAM
from models.resnet3D import resnet3D50


############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    #transform = Compose([CT_ndarrayToTensor(WW=1024, WL=0)])
    transform = Compose([Multi_CT_ndarrayToTensor(WW=[256,1024,100], WL=[0,0,50])])

    dataset = PVT_Loader_plot(root_dir=cfg.root_dir, length=cfg.length, dataset_type='train', transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=cfg.batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=2,
                                                pin_memory=True)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return loader

############## test関数 ##############
def plot_CAM(model, plot_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    Grad_CAM_3D = Mr_sota_haruki_Grad_CAM_3D(model, [model.layer[-1][-1]], device)
    CAM_3D = CAM(model, [model.layer[-1][-1]], device)

    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for batch_idx, (inputs, targets) in tqdm(enumerate(plot_loader), total=len(plot_loader), leave=False):
        #GPUモード高速化
        inputs = inputs.cuda(device, non_blocking=True)
        if targets['frame_id'] % 16 ==0:
            Grad_CAM_3D(inputs, targets)
            CAM_3D(inputs, targets)



@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")
    
    # モデル設定
    #model = resnet3D50(in_c=1, num_classes=2, non_local=True).cuda(device)
    model = resnet3D50(in_c=3, num_classes=2, non_local=True).cuda(device)

    # model load
    model_path = "result/model_test_acc.pth"
    #model_path = "result/model_train_loss.pth"
    model.load_state_dict(torch.load(model_path))


    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    #model = apex.parallel.convert_syncbn_model(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    loader = dataload(cfg.dataset)

    #高速化コードただし計算の再現性は無くなる
    #torch.backends.cudnn.benchmark = True

    # CAM
    plot_CAM(model, loader, device)


############## main ##############
if __name__ == '__main__':
    main()