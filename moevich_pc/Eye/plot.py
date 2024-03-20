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
from utils.dataset import (
    Eye_Loader,
    Eye_Loader_traintest,
    Eye_Loader_2classification2,
    Eye_Loader_2classification2_traintest)
from utils.visualization import CAM_2D
from models.resnet2D import resnet2D18, resnet2D50, resnet2D2_18, resnet2D2_50, resnet2D18_atten
from models.vit import ViT, ViT2class

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    transform = Compose([ndarrayToTensor(),
                                CenterCrop(size=(800,800)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    if cfg.class_ == 4:
        #dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='test', split=4, transform=transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
        dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='test', split=cfg.split, transform=transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    else:
        dataset = Eye_Loader_2classification2(root_dir=cfg.root_dir, dataset_type='test', split=cfg.split, transform=transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)

    #if cfg.class_ == 4:
    #    dataset = Eye_Loader_traintest(root_dir=cfg.root_dir, dataset_type='test', transform=transform)
    #else:
    #    dataset = Eye_Loader_2classification2_traintest(root_dir=cfg.root_dir, dataset_type='test', transform=transform)
    
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
    Grad_CAM_2D = CAM_2D(model, [model.layer[-1][-1]], device)
    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for batch_idx, (inputs, targets) in tqdm(enumerate(plot_loader), total=len(plot_loader), leave=False):
        Grad_CAM_2D(inputs, targets)


@hydra.main(config_path="config", config_name='plot.yaml')
def main(cfg):
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")
    
    in_channels = 3
    if cfg.dataset.use_age:
        in_channels += 1
    if cfg.dataset.use_sex:
        in_channels += 1
    
    # モデル設定
    if cfg.train_conf.model == "Resnet18":
        model = resnet2D18(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet50":
        model = resnet2D50(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "ViT":
        model = ViT(image_size = 800,
            patch_size = 32,
            num_classes = 4,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            ).cuda(device)
    if cfg.train_conf.model == "Resnet18_2":
        model = resnet2D2_18(in_c=in_channels, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet50_2":
        model = resnet2D2_50(in_c=in_channels, non_local=False).cuda(device)
    if cfg.train_conf.model == "ViT_2":
        model = ViT2class(image_size = 800,
            patch_size = 32,
            num_classes = 4,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            ).cuda(device)
    if cfg.train_conf.model == "Resnet18_atten":
        model = resnet2D18_atten(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    
    # model load
    model_path = "result/model_val_acc.pth"
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