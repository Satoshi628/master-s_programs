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
#from tsne_torch import TorchTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#----- My Module -----#
from utils.utils import ndarrayToTensor
from utils.dataset import (
    Eye_Loader,
    Eye_Loader_traintest,
    Eye_Loader_2classification2,
    Eye_Loader_2classification2_traintest)
from utils.visualization import CAM_2D
from models.resnet2D import resnet2D50, resnet2D18, resnet2D18_atten, resnet2D18_siam, resnet2D50_siam, resnet2D18_simCLR

from models.vit import ViT, ViT2class

############## dataloader関数##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    transform = Compose([ndarrayToTensor(),
                                CenterCrop(size=(800,800)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    train_dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='train', split=cfg.split, transform=transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    val_dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='val', split=cfg.split, transform=transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    test_dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='test', split=cfg.split, transform=transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            pin_memory=False)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            pin_memory=False)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2,
                                            pin_memory=False)
    
    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return train_loader, val_loader, test_loader

############## test関数 ##############
def plot_tSNE(model, train_loader, val_loader, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    train_feature = []
    val_feature = []
    test_feature = []

    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for inputs, targets in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        train_feature.append(model.get_feature(inputs).cpu().detach())
    
    for inputs, targets in tqdm(val_loader, total=len(val_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        val_feature.append(model.get_feature(inputs).cpu().detach())
    
    for inputs, targets in tqdm(test_loader, total=len(test_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        test_feature.append(model.get_feature(inputs).cpu().detach())
    
    train_feature = torch.cat(train_feature, dim=0)
    val_feature = torch.cat(val_feature, dim=0)
    test_feature = torch.cat(test_feature, dim=0)


    feature = torch.cat([train_feature, val_feature, test_feature], dim=0)
    
    colors = [
        *[[1., 0., 0.] for _ in range(train_feature.shape[0])],
        *[[0., 1., 0.] for _ in range(val_feature.shape[0])],
        *[[0., 0., 1.] for _ in range(test_feature.shape[0])]
    ]

    feature = feature.flatten(1, -1).cpu().numpy()

    #emb = PCA(n_components=2).fit_transform(feature)
    emb = TSNE(n_components=2, random_state=0).fit_transform(feature)

    plt.clf()
    for color, coord in zip(colors, emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"eye_image.png")


############## test関数 ##############
def plot_image_tSNE(model, train_loader, val_loader, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    train_inputs = []
    val_inputs = []
    test_inputs = []

    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for inputs, targets in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = F.avg_pool2d(inputs, 16)
        inputs = inputs[~torch.all(torch.all(inputs==inputs[:,:,0,0,None,None],dim=-1),dim=-1)]
        train_inputs.append(inputs)
    
    for inputs, targets in tqdm(val_loader, total=len(val_loader), leave=False):
        inputs = F.avg_pool2d(inputs, 16)
        val_inputs.append(inputs)
    
    for inputs, targets in tqdm(test_loader, total=len(test_loader), leave=False):
        inputs = F.avg_pool2d(inputs, 16)
        test_inputs.append(inputs)
    
    train_inputs = torch.cat(train_inputs, dim=0)
    val_inputs = torch.cat(val_inputs, dim=0)
    test_inputs = torch.cat(test_inputs, dim=0)


    #inputs = torch.cat([train_inputs[:-10], val_inputs[:-10], test_inputs[:-10]], dim=0)
    inputs = train_inputs


    #feature = val_feature
    
    # colors = [
    #     *[[1., 0., 0.] for _ in range(train_inputs.shape[0])],
    #     *[[0., 1., 0.] for _ in range(val_inputs.shape[0])],
    #     *[[0., 0., 1.] for _ in range(test_inputs.shape[0])]
    # ]
    colors = [
        *[[1., 0., 0.] for _ in range(int(train_inputs.shape[0]*0.15))],
        *[[0., 0., 1.] for _ in range(int(train_inputs.shape[0]*0.15))],
        *[[0.5, 0., 0.5] for _ in range(int(train_inputs.shape[0]*0.7))]
    ]

    inputs = inputs.flatten(1, -1).cpu().numpy()

    emb = TSNE(n_components=2, random_state=0).fit_transform(inputs)

    #pca = PCA(n_components=2)
    #pca.fit(inputs)
    #emb = pca.transform(inputs)
    plt.clf()
    for color, coord in zip(colors, emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"eye_image_image.png")

############## test関数 ##############
def class_tSNE(model, train_loader, val_loader, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    train_feature = []
    val_feature = []
    test_feature = []

    train_label = []
    val_label = []
    test_label = []
    
    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for inputs, targets in tqdm(train_loader, total=len(train_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        train_feature.append(model.get_feature(inputs).cpu().detach())
        train_label.append(targets)
    
    for inputs, targets in tqdm(val_loader, total=len(val_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        val_feature.append(model.get_feature(inputs).cpu().detach())
        val_label.append(targets)

    
    for inputs, targets in tqdm(test_loader, total=len(test_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        test_feature.append(model.get_feature(inputs).cpu().detach())
        test_label.append(targets)


    train_feature = torch.cat(train_feature,dim=0)
    val_feature = torch.cat(val_feature,dim=0)
    test_feature = torch.cat(test_feature,dim=0)

    train_label = torch.cat(train_label,dim=0)
    val_label = torch.cat(val_label,dim=0)
    test_label = torch.cat(test_label,dim=0)

    #feature = torch.cat([train_feature, val_feature, test_feature], dim=0)

    train_feature = train_feature.flatten(1, -1).cpu().numpy()
    val_feature = val_feature.flatten(1, -1).cpu().numpy()
    test_feature = test_feature.flatten(1, -1).cpu().numpy()
    
    train_emb = TSNE(n_components=2, random_state=0).fit_transform(train_feature)
    val_emb = TSNE(n_components=2, random_state=0).fit_transform(val_feature)
    test_emb = TSNE(n_components=2, random_state=0).fit_transform(test_feature)


    #colors = torch.tensor([[0.,0.,0.],[0.,0.,1.],[1.,0.,0.],[1.,0.,1.]])
    colors = torch.tensor([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
    
    train_colors = colors[train_label].tolist()
    val_colors = colors[val_label].tolist()
    test_colors = colors[test_label].tolist()

    plt.clf()
    for color, coord in zip(train_colors, train_emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)
    plt.savefig(f"eye_image_class_train.png")
    plt.clf()
    for color, coord in zip(val_colors, val_emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)
    plt.savefig(f"eye_image_class_val.png")
    plt.clf()
    for color, coord in zip(test_colors, test_emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)
    plt.savefig(f"eye_image_class_test.png")




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
    if cfg.train_conf.model == "Resnet18_siam":
        model = resnet2D18_siam(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet18_simCLR":
        model = resnet2D18_simCLR(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
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
    #model.load_state_dict(torch.load(model_path))


    #マルチGPU
    model = torch.nn.DataParallel(model) if cfg.train_conf.multi_gpu else model
    #model = apex.parallel.convert_syncbn_model(model) if cfg.train_conf.multi_gpu else model

    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    # データ読み込み+初期設定
    train_loader, val_loader, test_loader = dataload(cfg.dataset)

    #高速化コードただし計算の再現性は無くなる
    #torch.backends.cudnn.benchmark = True

    #plot_tSNE(model, train_loader, val_loader, test_loader, device)
    plot_image_tSNE(model, train_loader, val_loader, test_loader, device)
    #class_tSNE(model, train_loader, val_loader, test_loader, device)


############## main ##############
if __name__ == '__main__':
    main()