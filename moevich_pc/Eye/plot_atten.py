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
import matplotlib.pyplot as plt
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

def result(predict, label):

    if label.shape:
        return f"ARMS2:{predict[0]==label[0]}   CFH:{predict[1]==label[1]}"
    else:
        ARMS2 = f"{predict:02b}"[1] == f"{label:02b}"[1]
        CFH = f"{predict:02b}"[0] == f"{label:02b}"[0]
        return f"ARMS2:{ARMS2}   CFH:{CFH}"

############## test関数 ##############
def plot_CAM(model, plot_loader, device):
    # モデル→推論モード
    model.eval()
    alpha = 1.0
    colormap = plt.get_cmap('jet')
    image_num = 0
    upsamp = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    if not os.path.exists("attention_image"):
        os.mkdir("attention_image")

    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for batch_idx, (inputs, targets) in tqdm(enumerate(plot_loader), total=len(plot_loader), leave=False):
        inputs = inputs.cuda(device, non_blocking=True)
        outputs, atten = model(inputs)

        predict = outputs.max(dim=-1)[1].to("cpu").detach().numpy()

        atten = upsamp(atten)
        min_Imp = atten.flatten(-2).min(dim=-1)[0]
        max_Imp = atten.flatten(-2).max(dim=-1)[0]
        atten = (atten - min_Imp[:, None, None]) / (max_Imp[:, None, None] - min_Imp[:, None, None] + 1e-7)
        atten = atten.squeeze().cpu().detach().numpy()
        
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        images = inputs[:, :3] * std[None,:,None,None] + mean[None,:,None,None]
        images = (images * 255).cpu().detach().numpy().transpose(0, 2, 3, 1)

        for image, heatmap, pre_idx, label in zip(images, atten, predict, targets.to("cpu")):
            #heatmap.size => [b, H, W, 3]
            heatmap = (colormap(heatmap) * 255).astype(np.uint16)[:, :, :3]
            
            blended = image * alpha + heatmap * (1 - alpha)
            blended = blended.astype(np.uint8)
            plt.clf()
            plt.imshow(blended)
            plt.title(result(pre_idx, label))
            #メモリなし
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            #保存
            plt.savefig(f"attention_image/{image_num:05}.png",
                        bbox_inches='tight',
                        pad_inches=0.1)
            
            image_num += 1
            plt.close()



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