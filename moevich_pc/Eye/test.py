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
from torchvision.transforms import (Compose,
    Normalize,
    Resize,
    RandomCrop,
    CenterCrop,
    Pad,
    RandomHorizontalFlip,
    RandomVerticalFlip)
import torch.nn as nn
from tqdm import tqdm

#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.dataset import Eye_Loader, Eye_Loader_traintest
from models.resnet2D import resnet2D50, resnet2D18, resnet2D18_atten, resnet18_pretrained
from models.vit import ViT


############## dataloader関数##############
def dataload(cfg):
    test_transform = Compose([ndarrayToTensor(),
                                CenterCrop(size=(800,800)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    test_dataset = Eye_Loader(root_dir=cfg.root_dir, dataset_type='test', split=cfg.split, transform=test_transform, use_age=cfg.use_age, use_sex=cfg.use_sex, use_mask=cfg.use_mask)
    #test_dataset = Eye_Loader_traintest(root_dir=cfg.root_dir, dataset_type='test', transform=test_transform)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    #これらをやると
    #num_workers=os.cpu_count(),pin_memory=True
    return test_loader

############## test関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    total = 0
    correct1 = 0
    correct2 = 0
    correct_all = 0
    
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
            predict = output.max(dim=-1)[1]

            ###精度の計算###
            total += output.size(0)
            correct1 += (targets // 2 == predict // 2).sum()
            correct2 += (targets % 2 == predict % 2).sum()
            correct_all += ((targets// 2 == predict // 2) & (targets % 2 == predict % 2)).sum()

    return (correct1/total).item(), (correct2/total).item(), (correct_all/total).item()


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print(f"gpu         :{cfg.train_conf.gpu}")
    print(f"multi GPU   :{cfg.train_conf.multi_gpu}")

    ##### GPU設定 #####
    device = torch.device('cuda:{}'.format(cfg.train_conf.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")
    
    PATH = "result/test.txt"

    with open(PATH, mode='w') as f:
        f.write("test ARMS2 acc\ttest CFH acc\ttest all acc\n")
    
    in_channels = 3
    if cfg.dataset.use_age:
        in_channels += 1
    if cfg.dataset.use_sex:
        in_channels += 1
    
    #  モデル設定
    if cfg.train_conf.model == "Resnet18":
        model = resnet2D18(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "Resnet18_atten":
        model = resnet2D18_atten(in_c=in_channels, num_classes=4, non_local=False).cuda(device)
    if cfg.train_conf.model == "resnet18_pre":
        model = resnet18_pretrained(num_classes=4).cuda(device)
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
    test_loader = dataload(cfg.dataset)

    #高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    test_ARMS2_acc, test_CFH_acc, test_all_acc = test(model, test_loader, device)

    ##### 結果表示 #####
    result_text = f"test ARMS2 acc:{test_ARMS2_acc:.3%} \
    test CFH acc:{test_CFH_acc:.3%} \
    test ALL acc:{test_all_acc:.3%}"

    print(result_text)

    ##### 出力結果を書き込み #####
    with open(PATH, mode='a') as f:
        f.write(f"{test_ARMS2_acc:.3%}\t{test_CFH_acc:.3%}\t{test_all_acc:.3%}\n")
    


############## main ##############
if __name__ == '__main__':
    main()
