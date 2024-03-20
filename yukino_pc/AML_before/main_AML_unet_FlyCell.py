#coding: utf-8
##### ライブラリ読み込み ##########
import _common_function as cf
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import os
import argparse
import random
from tqdm import tqdm
from Net_AML_unet_FlyCell import Generator, Discriminator
from dataset import FlyCellDataLoader_crossval
import utils as ut

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time

#################################


class CrossEntropy_only(nn.Module):
    def __init__(self, device):
        super(CrossEntropy_only, self).__init__()
        self.ohv = cf.one_hot_vector(device, classification=5)

    def forward(self, G_img, inputs_b):

        #_, _, H, W = G_img.size()

        q = G_img.clone()
        p = inputs_b.clone()

        p = self.ohv.change(p)
        q = torch.log(q)
        p = p * q
        CrossEntropy = -p.mean()

        return CrossEntropy


class CrossEntropy_only_weight(nn.Module):
    def __init__(self, device, weights):
        super(CrossEntropy_only_weight, self).__init__()
        self.ohv = cf.one_hot_vector(device, classification=5)
        self.weights = weights

    def forward(self, G_img, inputs_b):
        #_, _, H, W = G_img.size()

        weights = self.weights.view(1, -1, 1, 1)

        q = G_img.clone()
        p = inputs_b.clone()

        p = self.ohv.change(p)
        q = torch.log(q)
        p = p * q
        Ans = weights * p
        CrossEntropy = -Ans.mean()

        return CrossEntropy


############## dataloader関数##############
def dataload():
    ds_train = FlyCellDataLoader_crossval(
        rootdir=args.rootdir, val_area=args.val_area, split='train', iteration_number=args.batchsize*args.iter)
    ds_val = FlyCellDataLoader_crossval(
        rootdir=args.rootdir, val_area=args.val_area, split='val')

    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batchsize, shuffle=True, num_workers=args.threads)
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batchsize, shuffle=False, num_workers=args.threads)

    return train_loader, val_loader


############## train関数 ##############
def train(epoch):
    # モデル→学習モード
    model_G.train()
    model_D.train()

    # 初期設定
    sum_loss_g = 0
    sum_loss_d = 0

    with torch.no_grad():

        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs_a, inputs_b) in enumerate(tqdm(train_loader, leave=False)):
            # GPUモード高速化
            inputs_a = inputs_a.cuda(device, non_blocking=True)
            inputs_b = inputs_b.cuda(device, non_blocking=True)

            ##### training generator #####

           # 入力画像をGeneratorに入力
            Segmentation_img = model_G(inputs_a, None, Train=False)

            # .detach() = 計算グラフを切る(やらないとメモリオーバーする)
            Segmentation_img_tensor = Segmentation_img[0].detach()

            # 入力画像と生成画像をPairでDiscriminatorに入力
            out = model_D(inputs_a, Segmentation_img, None, None, None)
            print('train',inputs_a.shape)
            # input()
            # print(Segmentation_img[0])
            # input()

            G_memory = [out[i].detach() for i in range(1, len(out))]

    for batch_idx, (inputs_a, inputs_b) in enumerate(tqdm(train_loader, leave=False)):
        # GPUモード高速化
        inputs_a = inputs_a.cuda(device, non_blocking=True)
        inputs_b = inputs_b.cuda(device, non_blocking=True)

        # print(inputs_b)

        #IOU_rabel = inputs_b
        #inputs_b = ohv.change(inputs_b)

        #inputs_a = inputs_a.long()
        #inputs_b = inputs_b.long()

        #print('G_memory=', G_memory[0].dtype)
        #print('inputs_a=', inputs_a.dtype)
        # print('inputs_b=',inputs_b.dtype)
        # input()

        # print(inputs_b.shape)  # torch.Size([8, 256, 256])
        # input()

        ##### training generator #####

        inputs_b = inputs_b.long()

        G_img, aux3, aux2, aux1, attention3, attention2, attention1 = model_G(
            inputs_a, G_memory, Train=True, gt=inputs_b)
        # print(G_img[0].shape)
        # input()

        # .detach() = 計算グラフを切る(やらないとメモリオーバーする)
        G_img_tensor = G_img.detach()

        D_out = model_D(inputs_a, G_img, attention3, attention2, attention1)

        ### 損失計算 ###
        lam_adv = 0.01
        # adversarial loss
        # print(G_img.max())
        loss_G_ad = lam_adv * criterion_ad(D_out[0], ones, use_Focal=True)

        loss_G_CE1 = criterion2(G_img, inputs_b)  # Softmax Cross Entropy Loss
        loss_G_CE2 = criterion2(aux3, inputs_b)  # Softmax Cross Entropy Loss
        loss_G_CE3 = criterion2(aux2, inputs_b)  # Softmax Cross Entropy Loss
        loss_G_CE4 = criterion2(aux1, inputs_b)   # Softmax Cross Entropy Loss
        loss_G = loss_G_ad + loss_G_CE1 + loss_G_CE2 + loss_G_CE3 + loss_G_CE4
        # print(G_img.max())
        # input()
        # 勾配を0
        model_D.zero_grad()
        model_G.zero_grad()

        # 誤差逆伝搬
        loss_G.backward()

        # パラメーター更新
        params_G.step()

        ##### training discriminator #####

        IOU_rabel = inputs_b
        inputs_b = ohv.change(inputs_b)

        # 入力画像と正解画像をPairでDiscriminatorに入力
        real_out = model_D(inputs_a, inputs_b, attention3,
                           attention2, attention1)

        # 入力画像と生成画像をPairでDiscriminatorに入力
        fake_out = model_D(inputs_a, G_img_tensor,
                           attention3, attention2, attention1)

        ### 損失計算 ###
        loss_D_real = criterion_ad(
            real_out[0], ones, use_Focal=False)  # adversarial loss
        loss_D_fake = criterion_ad(
            fake_out[0], zeros, use_Focal=False)  # adversarial loss
        loss_D = (loss_D_real + loss_D_fake) / 2.0

        # 勾配を0に
        model_D.zero_grad()
        model_G.zero_grad()

        # 誤差逆伝搬
        loss_D.backward()

        # パラメーター更新
        params_D.step()

        # loss溜め
        sum_loss_g += loss_G.item()
        sum_loss_d += loss_D.item()
        del loss_G
        del loss_D

        ###精度の計算###
        # print('G_img', G_img.size())  # G_img torch.Size([8, 4, 256, 256])
        # print('IOU_rabel', IOU_rabel.size()) #IOU_rabel torch.Size([8, 256, 256])

        IoU.collect(G_img, IOU_rabel, mode='linear')

    iou, miou = IoU.output()

    return sum_loss_g / (batch_idx + 1), sum_loss_d / (batch_idx + 1), iou, miou


############## val関数 ##############
def val(epoch):
    # モデル→学習モード
    model_G.eval()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル
        for batch_idx, (inputs_a, inputs_b) in enumerate(tqdm(val_loader, leave=False)):

            # GPUモード高速化
            inputs_a = inputs_a.cuda(device, non_blocking=True)
            print('val', inputs_a.shape)
            inputs_b = inputs_b.cuda(device, non_blocking=True)

            ##### generator #####

            # 入力画像をGeneratorに入力
            val_img = model_G(inputs_a, memory=None, Train=False)

            DI_out = model_D(inputs_a, val_img, None, None, None)

            val_img = model_G(inputs_a, DI_out[1:], Train=False)

            # print(inputs_a.size())
            # print(inputs_b.size())
            # print(val_img[0].size())
            # input()
            #torch.Size([16, 1, 256, 256])
            #torch.Size([16, 256, 256])
            #torch.Size([16, 5, 256, 256])

            # print(torch.min((val_img[0])))
            # input()

            ###精度の計算###
            IoU.collect(val_img, inputs_b, mode='linear')

        iou, miou = IoU.output()

    return iou, miou


############## main ##############
if __name__ == '__main__':

    start_time = time.time()

    ####### コマンド設定 #######
    parser = argparse.ArgumentParser(description='FlyCell')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=0)
    # ミニバッチサイズ(学習)指定
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    # ミニバッチサイズ(学習)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4)

    parser.add_argument("--rootdir", type=str,
                        default='/media/hykw/data/Kamiya/code/AML_before/data')

    # iter : イテレーション数(1epoch内で何回ランダムクロップを繰り返すか)
    parser.add_argument("--iter", default=12, type=int)

    #threads : num_workers数;何個GPUを使っているか
    parser.add_argument("--threads", default=2, type=int)

    # val_area : validationのブロック番号(１～５を選択). これを変えることで5回のCross Validationが可能
    parser.add_argument("--val_area", '-v', type=int, default=1,
                        help='cross-val test area [default: 5]')

    # 学習回数(epoch)指定
    parser.add_argument('--num_epochs', '-e', type=int, default=400)
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result')
    # lamda指定
    parser.add_argument('--lamda', '-lm', type=float, default=100)
    # 乱数指定
    parser.add_argument('--seed', '-s', type=int, default=0)

    args = parser.parse_args()

    ##### 初期設定表示 #####

    print("[Experimental conditions]")
    print(" GPU ID         : {}".format(args.gpu))
    print(" Epochs         : {}".format(args.num_epochs))
    print(" Minibatch size : {}".format(args.batchsize))
    print("")

    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    ##### 保存ディレクトリ・ファイル #####

    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))

    PATH_1 = "{}/loss_G.txt".format(args.out)
    PATH_2 = "{}/loss_D.txt".format(args.out)
    PATH_3 = "{}/train_IoU.txt".format(args.out)
    PATH_4 = "{}/val_IoU.txt".format(args.out)

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        pass
    with open(PATH_3, mode='w') as f:
        pass
    with open(PATH_4, mode='w') as f:
        pass

    ##### Covid-19 dataset #####

    classes = 5   # class number
    inputs_ch = 1  # input channel

    # モデル設定
    model_G = Generator(n_channels=inputs_ch,
                        n_classes=classes, device=device, bilinear=True).cuda(device)
    model_D = Discriminator(in_ch=inputs_ch+classes, out_ch=5).cuda(device)
    # in_ch=inputs_ch+classespairでDisににinputするから

    model_G = torch.nn.DataParallel(model_G)
    model_D = torch.nn.DataParallel(model_D)
    image = cf.P2P_Image()

    # IoUクラス定義
    IoU = cf.IoU(classification=5)

    # one_hot_vectorクラス定義
    ohv = cf.one_hot_vector(device, classification=5)

    # 損失関数設定

    criterion_ad = cf.P2P_CrossEntoropyLoss()  # Binaly Corss Entropy Loss

    #weights = torch.tensor([25.0, 25.0, 25.0, 25.0]).cuda(device)
    #criterion1 = CrossEntropy_only_weight(device, weights=weights)
    criterion2 = CrossEntropy_only(device)
    # optimizer設定

    params_G = torch.optim.Adam(
        model_G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    params_D = torch.optim.Adam(
        model_D.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # 初期値の乱数設定
    random.seed(11)  # python
    np.random.seed(11)  # numpy
    torch.manual_seed(11)  # pytorch

    # データ読み込み
    train_loader, val_loader = dataload()

    torch.backends.cudnn.benchmark = True

    ones = torch.ones(args.batchsize, 1, 16, 16).cuda(device)
    zeros = torch.zeros(args.batchsize, 1, 16, 16).cuda(device)

    ##### training & validation #####
    best_mIoU = 0.
    best_loss = 1000

    G_loss = []
    D_loss = []
    Train_mIoU = []
    Val_mIoU = []

    for epoch in range(args.num_epochs):

        train_loss_g, train_loss_d, _, train_miou = train(epoch)  # train
        _, val_miou = val(epoch)

        G_loss.append(train_loss_g)
        D_loss.append(train_loss_d)
        Train_mIoU.append(train_miou * 100)
        Val_mIoU.append(val_miou * 100)

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(G_loss, label="G_loss")
        plt.plot(D_loss, label="D_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        img_name = "loss_graph"
        plt.savefig(
            '/mnt/addhdd/Usha/AML_unet_FlyCell/{}/A_{}_loss_Graph.png'.format(args.out, args.out))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation mIoU")
        Tm = plt.plot(Train_mIoU, label="Train_mIoU")
        Vm = plt.plot(Val_mIoU, label="Val_mIoU")
        plt.xlabel("Epoch")
        plt.ylabel("mIoU[%]")
        plt.legend()
        plt.show()
        img_name = "mIoU_graph"
        # Tm.ylabel.set_major_formatter(mtick.PercentFormatter(
        #    xmax=1, decimals=None, symbol='%', is_latex=False))

        plt.savefig(
            '/mnt/addhdd/Usha/AML_unet_FlyCell/{}/A_{}_mIoU_Graph.png'.format(args.out, args.out))
        plt.close()

        ##### 結果表示 #####
        print("Epoch{:3d}/{:3d}  LossG={:.4f}  LossD={:.4f} train_miou={:.2f}%  val_miou={:.2f}%".format(epoch + 1,
                                                                                                         args.num_epochs,
                                                                                                         train_loss_g,
                                                                                                         train_loss_d,
                                                                                                         train_miou * 100,
                                                                                                         val_miou * 100))

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write("{},{:.2f}\n".format(epoch + 1, train_loss_g))
        with open(PATH_2, mode='a') as f:
            f.write("{},{:.2f}\n".format(epoch+1, train_loss_d))
        with open(PATH_3, mode='a') as f:
            f.write("{},{:.2f}\n".format(epoch+1, train_miou*100))
        with open(PATH_4, mode='a') as f:
            f.write("{},{:.2f}\n".format(epoch + 1, val_miou * 100))

        if val_miou >= best_mIoU:
            best_mIoU = val_miou
            PATH = "{}/model_D_IoU.pth".format(args.out)
            torch.save(model_D.state_dict(), PATH)
            PATH = "{}/model_G_IoU.pth".format(args.out)
            torch.save(model_G.state_dict(), PATH)

        if train_loss_g <= best_loss:
            best_loss = train_loss_g
            PATH = "{}/model_D_loss.pth".format(args.out)
            torch.save(model_D.state_dict(), PATH)
            PATH = "{}/model_G_loss.pth".format(args.out)
            torch.save(model_G.state_dict(), PATH)

    print("最高mIoU:{:.2f}%".format(best_mIoU * 100))

    end_time = time.time()
    print('Train_Time_in_min:', (end_time - start_time)/60)
