#coding: utf-8
#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep

#----- 専用ライブラリ -----#
import tqdm
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import (Compose,
    Normalize,
    Resize,
    RandomCrop,
    CenterCrop,
    ColorJitter,
    Pad,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomPerspective,
    RandomRotation)
import timm
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import cv2

#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_OCT_VIT_MASK
from models.resnet2D import resnet2D18, resnet2D18_siam, resnet2D50, resnet2D50_siam, resnet18_pretrained, resnet50_pretrained
from utils.loss import Contrastive_Loss, FocalLoss
from utils.evaluation import Precition_Recall

############## dataloader関数##############
def dataload(split):
    val_transform = Compose([ndarrayToTensor(),
                                Resize([315,315]),
                                CenterCrop(size=(288,288)),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    test_dataset = Eye_Loader_OCT_VIT_MASK(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='test', split=split, transform=val_transform,
                    use_age=False,
                    use_sex=False,
                    use_smoke=False,
                    use_AMD=True,
                    use_mask="HE"
                    )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    # これらをやると
    # num_workers=os.cpu_count(),pin_memory=True
    return test_loader

def threshold(array, p=0.9):
    uni_num = np.unique(array)

    error = 9999
    for uni in uni_num:
        m_p = (array < uni).sum() / array.shape
        if error > abs(m_p - p):
            error = abs(m_p - p)
            th = uni

    return th

############## val関数 ##############
def plot(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    num_list = []
    area_list = []
    HW_list = []
    aspect_list = []
    areas_list = []

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (_, _, mask, _, _) in enumerate(tqdm.tqdm(test_loader)):

            mask = mask[:, 1]
            mask = mask.numpy().astype(np.uint8)
            for m in mask:
                retval, _, stats, centroids = cv2.connectedComponentsWithStats(m)
                stats = stats[1:]

                num_list.append(retval - 1)
                area_list.append(stats[:, -1].sum())
                areas_list.append(stats[:, -1])
                HW_list.append(stats[:,2] * stats[:,3])
                aspect_list.append(stats[:,2] / stats[:,3])

    num_list = np.array(num_list)
    areas_list = np.concatenate(areas_list)
    HW_list = np.concatenate(HW_list)
    aspect_list = np.concatenate(aspect_list)

    print(HW_list.shape)
    print(aspect_list.shape)

    HE_acc = 0.9868
    #HE精度が中心確率の正規分布時の両端のx値
    normal_x = 1.857781847089388868599
    SE_acc = 0.9712
    #SE精度が中心確率の正規分布時の両端のx値
    # normal_x = 1.65069925516084616523

    aspect_mean = aspect_list.mean()
    aspect_std = aspect_list.std()
    print(aspect_mean)
    print(aspect_std)
    
    aspectX = aspect_std * normal_x + aspect_mean
    aspect_X = aspect_std * -normal_x + aspect_mean
    print("aspect <th:", aspectX)
    print("aspect >th:", aspect_X)

    HW_th = threshold(HW_list, 0.99)
    areas_th = threshold(areas_list, 0.99)
    print("H*W th:", HW_th)

    n, _, _ = plt.hist(HW_list, bins=100, range=[0,3000], alpha = 0.5, color="b")
    plt.vlines(HW_th, 0, max(n), "k")
    
    plt.savefig(f"HW_hist_line.png")
    plt.cla()
    plt.close()
    plt.clf()
    # n, _, _ = plt.hist(HW_list, bins=100, range=[0, 1000], alpha = 0.5, color="b")
    # plt.savefig(f"HW_hist0~1000.png")
    # plt.cla()
    # plt.close()
    # plt.clf()

    n, _, _ = plt.hist(aspect_list, bins=100, range=[0,2.], alpha = 0.5, color="b")
    plt.savefig(f"aspe_hist.png")
    plt.cla()
    plt.close()
    plt.clf()
    n, _, _ = plt.hist(aspect_list, bins=100, range=[0,2.], alpha = 0.5, color="b")
    plt.vlines(aspectX, 0, max(n), "k")
    plt.vlines(aspect_X, 0, max(n), "k")
    plt.savefig(f"aspe_hist_line.png")
    plt.cla()
    plt.close()
    plt.clf()

    n, _, _ = plt.hist(areas_list, bins=100, alpha = 0.5, color="b")
    plt.vlines(areas_th, 0, max(n), "k")
    
    plt.savefig(f"area_hist_line.png")
    plt.cla()
    plt.close()
    plt.clf()
    
    corrects = []
    num_list = []
    area_list = []

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (images, OCT, mask, targets, info) in enumerate(test_loader):
            # GPUモード高速化
            inputs = images.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output = model(inputs)
            _, idx = output.max(dim=-1)
            idx = idx//2
            targets = targets//2
            correct = ((idx == targets)*1).long()

            corrects.append(correct.cpu().numpy())

            #HEからSE部分を削除
            #0gaHE, 1gaSE
            # mask = mask[:, 0] * (mask[:, 1] == 0)
            mask = mask[:, 0]
            mask = mask.numpy().astype(np.uint8)
            for idx, m in enumerate(mask):
                retval, _, stats, centroids = cv2.connectedComponentsWithStats(m)
                stats = stats[1:]
                
                flag = (stats[:, -1] < 10) | (stats[:,2] * stats[:,3] > HW_th) | (stats[:,2] / stats[:,3] > aspectX) | (stats[:,2] / stats[:,3] < aspect_X)
                # flag = stats[:,-1] == -10

                num_list.append(retval - 1 - flag.sum())
                area_list.append(stats[~flag, -1].sum())

                bbox = stats[~flag, :4]
                bbox[:, 2:] = bbox[:, :2] + bbox[:, 2:]
                m = torch.from_numpy(m[None]).expand(3,-1,-1)

                img = torchvision.utils.draw_bounding_boxes(m*255, torch.from_numpy(bbox), colors=[(255,0,0) for _ in bbox])
                torchvision.utils.save_image(img[None].float(), f"bbox_{batch_idx:02}_{idx:02}.png", normalize=True)


    
    corrects = np.concatenate(corrects, axis=0)
    num_list = np.array(num_list)
    print(num_list)
    area_list = np.array(area_list)

    #nums
    plt.hist(num_list[num_list > 0], bins=20)
    plt.savefig(f"nums_hist.png")
    plt.cla()
    plt.close()
    plt.clf()

    print("0~10個", corrects[(num_list > 0) & (num_list < 10)].mean(), corrects[(num_list > 0) & (num_list < 10)].shape[0])
    print("10~20個", corrects[(num_list > 10) & (num_list < 20)].mean(), corrects[(num_list > 10) & (num_list < 20)].shape[0])
    print("20~30個", corrects[(num_list > 20) & (num_list < 30)].mean(), corrects[(num_list > 20) & (num_list < 30)].shape[0])
    print("30~60個", corrects[(num_list > 30) & (num_list < 60)].mean(), corrects[(num_list > 30) & (num_list < 60)].shape[0])
    print("60~90個", corrects[(num_list > 60) & (num_list < 90)].mean(), corrects[(num_list > 60) & (num_list < 90)].shape[0])
    print("90個以上", corrects[num_list > 90].mean(), corrects[num_list > 90].shape[0])

    #hist
    plt.hist(area_list[area_list > 0], bins=20)
    plt.savefig(f"area_hist.png")
    plt.cla()
    plt.close()
    plt.clf()

    print("総面積100以下", corrects[(area_list > 0) & (area_list < 100)].mean(), corrects[(area_list > 0) & (area_list < 100)].shape[0])
    print("総面積100~300", corrects[(area_list > 100) & (area_list < 300)].mean(), corrects[(area_list > 100) & (area_list < 300)].shape[0])
    print("総面積300~500", corrects[(area_list > 300) & (area_list < 500)].mean(), corrects[(area_list > 300) & (area_list < 500)].shape[0])
    print("総面積500~700", corrects[(area_list > 500) & (area_list < 700)].mean(), corrects[(area_list > 500) & (area_list < 700)].shape[0])
    print("総面積700~900", corrects[(area_list > 700) & (area_list < 900)].mean(), corrects[(area_list > 700) & (area_list < 900)].shape[0])
    print("総面積900~", corrects[area_list > 900].mean(), corrects[area_list > 900].shape[0])


############## val関数 ##############
def check(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    corrects = []
    num_list = []
    area_list = []
    areas_list = []
    mean = torch.tensor([0.485, 0.456, 0.406],device=device)[None,:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225],device=device)[None,:,None,None]

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, mask, targets, info) in enumerate(test_loader):
            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs_OCT = torch.cat([inputs, OCT], dim=-2)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output = model(inputs_OCT)
            _, idx = output.max(dim=-1)
            idx = idx//2
            targets = targets//2
            correct = ((idx == targets)*1).long()

            corrects.append(correct.cpu().numpy())
            inputs = std*inputs + mean
            torchvision.utils.save_image(inputs, f"images/img{batch_idx:04}.png")
            mask = mask[:, 0] * (mask[:, 1] == 0)
            # torchvision.utils.save_image(mask[:, None].float(), f"images/HE_noSE_mask{batch_idx:04}.png")
            mask = mask.numpy().astype(np.uint8)
            for m in mask:
                retval, _, stats, centroids = cv2.connectedComponentsWithStats(m)
                stats = stats[1:]
                flag = (stats[:, -1] < 10) | (stats[:,2] * stats[:,3] > 1000)

                num_list.append(retval - 1 - flag.sum())
                area_list.append(stats[~flag, -1].sum())
                areas_list.append(stats[~flag, -1])
    
    corrects = np.concatenate(corrects, axis=0)
    num_list = np.array(num_list)
    area_list = np.array(area_list)
    areas_list = np.array(areas_list)

    print(area_list)
    print(areas_list)
    print(area_list.shape)
    print(areas_list.shape)



############## val関数 ##############
def mask_soukan(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    corrects = []
    num_list = []
    area_list = []
    areas_list = []

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, OCT, mask, targets, info) in enumerate(test_loader):
        # GPUモード高速化
        corrects.append(targets.numpy())
        mask = mask.numpy().astype(np.uint8)

        for m in mask:
            retval, _, stats, centroids = cv2.connectedComponentsWithStats(m)
            num_list.append(retval - 1)
            stats = stats[1:]
            area_list.append(stats[:, -1].sum())
            areas_list.append(stats[:, -1])
    
    corrects = np.concatenate(corrects, axis=0)
    num_list = np.array(num_list)
    area_list = np.array(area_list)
    ARMS2_targets = corrects // 2

    data = [num_list[ARMS2_targets == 0], num_list[ARMS2_targets == 1]]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data)  # 複数指定する場合はタプル型で渡します。
    ax.set_xticklabels(['No ARMS2', 'ARMS2'])
    ax.set_ylim(0, 500)
    plt.grid()  # 横線ラインを入れることができます。
    plt.savefig(f"mask箱ひげ_nums.png")
    plt.cla()
    plt.close()
    plt.clf()

    data = [area_list[ARMS2_targets == 0], area_list[ARMS2_targets == 1]]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data)  # 複数指定する場合はタプル型で渡します。
    ax.set_xticklabels(['No ARMS2', 'ARMS2'])
    ax.set_ylim(0, 10000)
    plt.grid()  # 横線ラインを入れることができます。
    plt.savefig(f"mask箱ひげ_area.png")
    plt.cla()
    plt.close()
    plt.clf()

############## test関数 ##############
def plot_image_tSNE(model, test_loader, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    test_inputs = []

    for inputs, OCT, targets, age, sex in tqdm(test_loader, total=len(test_loader), leave=False):
        inputs = F.avg_pool2d(inputs, 16)
        test_inputs.append(inputs)
    
    test_inputs = torch.cat(test_inputs, dim=0)
    
    colors = [
        *[[1., 0., 0.] for _ in range(train_inputs.shape[0])],
        *[[0., 1., 0.] for _ in range(val_inputs.shape[0])],
        *[[0., 0., 1.] for _ in range(test_inputs.shape[0])]
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


@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    test_loader = dataload(cfg.dataset.split)
    device = torch.device('cuda:0')

    #model_name = "vit_tiny_patch16_224_in21k"
    model_name = "vit_large_patch32_224_in21k"
    pretrained = False
    model = timm.create_model(model_name, img_size=[576,288], pretrained=pretrained, num_classes=4)

    # model load
    model_path = "model_val_acc.pth"
    model.load_state_dict(torch.load(model_path))

    model = model.cuda(device)

    plot(model, test_loader, device)
    # mask_soukan(model, test_loader, device)
    # check(model, test_loader, device)
    # test(model, test_loader, device)

############## main ##############
if __name__ == '__main__':
    # 初期値の乱数設定
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # 高速化コードただし計算の再現性は無くなる
    torch.backends.cudnn.benchmark = True

    main()
