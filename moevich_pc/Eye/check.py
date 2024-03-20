#coding: utf-8
#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
from time import sleep

#----- 専用ライブラリ -----#
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
import matplotlib.pyplot as plt

#----- 自作モジュール -----#
from utils.utils import ndarrayToTensor
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Eye_Loader_OCT_VIT, Eye_Loader_OCT_VIT_CHECK
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

    test_dataset = Eye_Loader_OCT_VIT_CHECK(root_dir="/mnt/hdd1/kamiya/dataset/Eye_edit", dataset_type='test', split=split, transform=val_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=2)

    # これらをやると
    # num_workers=os.cpu_count(),pin_memory=True
    return test_loader


############## val関数 ##############
def test(model, test_loader, device):
    # モデル→推論モード
    model.eval()
    
    
    # 初期設定
    set_color = [[1., 0., 0.],[0., 0., 0.]]
    colors = []
    inputs_img = []
    inputs_OCT = []
    age_list = []
    sex_list = []
    output_list = []
    targets_list = []
    attention_list = []
    man_corre = 0.
    man_all = 0.
    woman_corre = 0.
    woman_all = 0.
    #50以下,50代,60代,70代,80以上
    age_sec = [0,50,60,70,80,1000]
    age_corre = [0.,0.,0.,0.,0.]
    age_all = [0.,0.,0.,0.,0.]
    

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, targets, age, sex) in enumerate(test_loader):
            
            inputs_img.append(inputs)
            inputs_OCT.append(OCT.float())
            age_list.append(age)
            sex_list.append(sex)

            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs)
            output_list.append(output.to("cpu"))
            attention_list.append(attens[2].to("cpu"))
            targets_list.append(targets.to("cpu"))
            _,idx = output.max(dim=-1)
            idx = idx//2
            targets = targets//2
            idx = ((idx == targets)*1).long()
            
            man_corre += idx.to("cpu")[sex==0].sum()
            man_all += (sex==0).sum()
            woman_corre += idx.to("cpu")[sex==1].sum()
            woman_all += (sex==1).sum()
            for i in range(len(age_corre)):
                flag = (age_sec[i] < age) & (age_sec[i+1] >= age)
                age_corre[i] += idx.to("cpu")[flag].sum()
                age_all[i] += (flag*1).sum()

            colors.extend([set_color[i] for i in idx])
    
    print(man_all)
    print(woman_all)
    print("man accuracy")
    print(man_corre/man_all)
    print("woman accuracy")
    print(woman_corre/woman_all)

    for i in range(len(age_corre)):
        print(f"{age_sec[i]}-{age_sec[i+1]}")
        print(age_corre[i]/age_all[i])
        print(age_all[i])

    inputs_img = torch.cat(inputs_img, dim=0)
    inputs_OCT = torch.cat(inputs_OCT, dim=0)
    output_list = torch.cat(output_list, dim=0)
    attention_list = torch.cat(attention_list, dim=0)
    targets_list = torch.cat(targets_list, dim=0)
    age_list = torch.cat(age_list, dim=0)
    sex_list = torch.cat(sex_list, dim=0)
    
    inputs_img = inputs_img.flatten(1, -1).cpu().numpy()
    inputs_OCT = inputs_OCT.flatten(1, -1).cpu().numpy()
    print(attention_list.shape)
    atten_img = attention_list.sum(dim=1)
    atten_img = (atten_img - atten_img.min())/(atten_img.max() - atten_img.min()+1e-7)

    attention_list = attention_list.flatten(1, -1).cpu().numpy()

    emb = TSNE(n_components=2, random_state=0).fit_transform(inputs_img)

    plt.clf()
    for color, coord in zip(colors, emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"img_tSNE.png")
    plt.cla()
    plt.close()
    plt.clf()

    emb = TSNE(n_components=2, random_state=0).fit_transform(inputs_OCT)
    for color, coord in zip(colors, emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"OCT_tSNE.png")
    plt.cla()
    plt.close()

    #年齢箱ひげ図
    non_data = age_list[(targets_list//2) == 0].to("cpu").numpy()
    true_data = age_list[(targets_list//2) == 1].to("cpu").numpy()
    fig, ax = plt.subplots()

    bp = ax.boxplot([non_data, true_data]) # 複数指定する場合はタプル型で渡します。
    ax.set_xticklabels(['non ARMS2', 'ARMS2'])
    plt.grid() # 横線ラインを入れることができます。

    plt.savefig(f"targets_agebox.png")
    plt.cla()
    plt.close()
    plt.clf()

    fig, ax = plt.subplots()

    bp = ax.boxplot([non_data[:-10], true_data[:-10]]) # 複数指定する場合はタプル型で渡します。
    ax.set_xticklabels(['non ARMS2', 'ARMS2'])
    plt.grid() # 横線ラインを入れることができます。

    plt.savefig(f"outputs_agebox.png")
    plt.cla()
    plt.close()
    plt.clf()

    emb = TSNE(n_components=2, random_state=0).fit_transform(attention_list)
    for color, coord in zip(colors, emb):
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"atten_tSNE.png")
    plt.cla()
    plt.close()
    left = atten_img[np.argmax(emb[:,0])]
    right = atten_img[np.argmin(emb[:,0])]
    top = atten_img[np.argmax(emb[:,1])]
    bottom = atten_img[np.argmin(emb[:,1])]
    plt.imshow(left)
    plt.savefig(f"left.png")
    plt.cla()
    plt.close()

    plt.imshow(right)
    plt.savefig(f"right.png")
    plt.cla()
    plt.close()
    
    plt.imshow(top)
    plt.savefig(f"top.png")
    plt.cla()
    plt.close()
    
    plt.imshow(bottom)
    plt.savefig(f"bottom.png")
    plt.cla()
    plt.close()



############## val関数 ##############
def plot(model, test_loader, device):
    # モデル→推論モード
    model.eval()
    
    
    # 初期設定
    set_color = [[1., 0., 0.],[0., 0., 0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.]]
    colors = []
    inputs_img = []
    inputs_OCT = []
    age_list = []
    sex_list = []
    output_list = []
    targets_list = []
    attention_list = []
    man_corre = 0.
    man_all = 0.
    woman_corre = 0.
    woman_all = 0.
    #50以下,50代,60代,70代,80以上
    age_sec = [0,50,60,70,80,1000]
    age_corre = [0.,0.,0.,0.,0.]
    age_all = [0.,0.,0.,0.,0.]
    

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs, OCT, targets, age, sex) in enumerate(test_loader):
            flag = ~torch.all(torch.all(OCT==OCT[:,:,0,0,None,None],dim=-1),dim=-1)[:,0]
            inputs = inputs[flag]
            OCT = OCT[flag]
            targets = targets[flag]
            age = age[flag]
            sex = sex[flag]
            
            inputs_img.append(inputs)
            inputs_OCT.append(OCT.float())
            age_list.append(age)
            sex_list.append(sex)

            # GPUモード高速化
            inputs = inputs.cuda(device, non_blocking=True)
            OCT = OCT.float().cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            # H方向に画像を結合
            inputs = torch.cat([inputs, OCT], dim=-2)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, attens = model(inputs)
            output_list.append(output.to("cpu"))
            attention_list.append(attens[2].to("cpu"))
            targets_list.append(targets.to("cpu"))
            _,idx = output.max(dim=-1)
            idx = idx//2
            targets = targets//2
            idx = ((idx == targets)*1).long()
            
            man_corre += idx.to("cpu")[sex==0].sum()
            man_all += (sex==0).sum()
            woman_corre += idx.to("cpu")[sex==1].sum()
            woman_all += (sex==1).sum()
            for i in range(len(age_corre)):
                flag = (age_sec[i] < age) & (age_sec[i+1] >= age)
                age_corre[i] += idx.to("cpu")[flag].sum()
                age_all[i] += (flag*1).sum()

    
    print(man_all)
    print(woman_all)
    print("man accuracy")
    print(man_corre/man_all)
    print("woman accuracy")
    print(woman_corre/woman_all)
    
    for i in range(len(age_corre)):
        print(f"{age_sec[i]}-{age_sec[i+1]}")
        print(age_corre[i]/age_all[i])
        print(age_all[i])

    inputs_img = torch.cat(inputs_img, dim=0)
    inputs_OCT = torch.cat(inputs_OCT, dim=0)
    targets_list = torch.cat(targets_list, dim=0)
    age_list = torch.cat(age_list, dim=0)
    sex_list = torch.cat(sex_list, dim=0)

    age_num = torch.zeros(age_list.shape,dtype=torch.long)
    age_num[age_list < 50] = 0
    age_num[(age_list < 60) & (50 <= age_list)] = 1
    age_num[(age_list < 70) & (60 <= age_list)] = 2
    age_num[(age_list < 80) & (70 <= age_list)] = 3
    age_num[age_list > 80] = 4

    print((targets_list[age_num==0]//2).float().mean())
    print((targets_list[age_num==1]//2).float().mean())
    print((targets_list[age_num==2]//2).float().mean())
    print((targets_list[age_num==3]//2).float().mean())
    print((targets_list[age_num==4]//2).float().mean())
    print((targets_list[sex_list==1]//2).float().mean())
    print((targets_list[sex_list==0]//2).float().mean())
    print((age_num==0).sum())
    print((age_num==1).sum())
    print((age_num==2).sum())
    print((age_num==3).sum())
    print((age_num==4).sum())
    print((sex_list==1).sum()) #男性
    print((sex_list==0).sum()) #女性

    
    inputs_img = inputs_img.flatten(1, -1).cpu().numpy()
    inputs_OCT = inputs_OCT.flatten(1, -1).cpu().numpy()
    emb = TSNE(n_components=2, random_state=0).fit_transform(inputs_img)

    plt.clf()
    for age_n, coord in zip(age_num, emb):
        color = set_color[age_n.item()]
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"img_tSNE.png")
    plt.cla()
    plt.close()
    plt.clf()

    
    emb = TSNE(n_components=2, random_state=0).fit_transform(inputs_OCT)
    for age_n, coord in zip(age_num, emb):
        color = set_color[age_n.item()]
        plt.plot(coord[0], coord[1], marker=".", color=color)

    plt.savefig(f"OCT_tSNE.png")
    plt.cla()
    plt.close()




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
