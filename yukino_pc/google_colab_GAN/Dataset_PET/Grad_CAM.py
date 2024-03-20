#coding: utf-8
#----- 標準ライブラリ -----#
import os

#----- 専用ライブラリ -----#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

#----- 自作モジュール -----#
from Dataset_PET.dataloader import ANIMAL_NAME_JA,ANIMAL_NAME_EN



############## Grad CAM 関数 ##############
def Grad_CAM(model, data_loader, out_file_path):
    if not os.path.exists(out_file_path + "/image"):
        os.mkdir(out_file_path + "/image")
    
    #保存用番号作成
    save_num = {animal:0 for animal in ANIMAL_NAME_EN}

    #device設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # モデル→推論モード
    model.cuda(device)
    model.eval()

    # 学習ループ  inputs:入力画像  targets:教師ラベル画像
    for inputs, targets in data_loader:
        # GPUモード
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        #入力画像をモデルに入力
        #predict.size()=>[batch,class num] feature.size()=>[batch,channel,H,W]
        _, feature = model(inputs)  # 特徴マップを計算

        #featureはleaf Tensorにしなければならない
        feature = feature.clone().detach().requires_grad_(True)

        predict = model.feature_to_class(feature)
        class_value, _ = predict.max(dim=-1)
        
        model.zero_grad()
        # 予測でもっとも高い値をとったクラスの勾配を計算
        #batchがある場合、合計してから計算
        class_value.sum().backward(retain_graph=False)

        # 特徴量マップ重み[batch,channel]
        alpha = feature.grad.mean(dim=(-2, -1))
        
        #重要度マップ作成
        alpha = F.relu(alpha)
        Imp_map = feature * alpha[:, :, None, None] #[batch,channel,H,W]
        Imp_map = Imp_map.sum(dim=1)  #[batch,H,W]
        
        #[0,1]になるようにmax-min正規化
        min_Imp = Imp_map.flatten(-2).min(dim=-1)[0]
        max_Imp = Imp_map.flatten(-2).max(dim=-1)[0]
        Imp_map = (Imp_map - min_Imp[:, None, None]) / (max_Imp[:, None, None] - min_Imp[:, None, None] + 1e-7)
        
        #Numpy化、inputsのサイズにリサイズ
        upsamp = nn.Upsample(inputs.shape[-2], mode="bilinear", align_corners=True)
        #bilinearモードは4次元出なくてはならないため
        Imp_map = upsamp(Imp_map[:, None]).squeeze(1)
        Imp_map = Imp_map.cpu().detach().numpy()

        #入力画像編集
        #Normalizeしているため元に戻す
        inputs = inputs * 0.5 + 0.5
        images = (inputs * 255).cpu().detach().numpy().transpose(0, 2, 3, 1)
        
        #確信度など編集
        predict = F.softmax(predict, dim=-1)
        class_value, predict_idx = predict.max(dim=-1)

        targets = targets.cpu().detach().numpy()
        class_value = class_value.cpu().detach().numpy()
        predict_idx = predict_idx.cpu().detach().numpy()


        colormap = plt.get_cmap('jet')
        alpha = 0.7
        #batch処理ができないためここからfor文
        for image, heatmap, value, label, pre_idx in zip(images, Imp_map, class_value, targets, predict_idx):
            heatmap = (colormap(heatmap) * 255).astype(np.uint16)[:, :, :3]
            
            blended = image * alpha + heatmap * (1 - alpha)
            blended = blended.astype(np.uint8)

            plt.clf()
            plt.figure(figsize=(10, 10))
            #目盛りなし
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.title(f"{ANIMAL_NAME_EN[pre_idx]}:{value:3.2%}", fontsize=24)
            plt.imshow(blended)

            plt.savefig(out_file_path + "/image" +
                        f"/{ANIMAL_NAME_JA[label]}_{save_num[ANIMAL_NAME_EN[label]]:04}.png", dpi=150
                        , bbox_inches='tight', pad_inches=0.1)
            plt.close()

            if save_num[ANIMAL_NAME_EN[label]] == 0:
                #matplotでは表示+保存の相性が悪い。
                img = Image.open(out_file_path + "/image" + f"/{ANIMAL_NAME_JA[label]}_{save_num[ANIMAL_NAME_EN[label]]:04}.png")
                # 規定のソフトで画像を閲覧
                plt.clf()
                plt.imshow(img)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                # 上下左右の余白削除
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                # グラフの枠も削除
                plt.box(False)
                plt.show()
                plt.close()

            
            save_num[ANIMAL_NAME_EN[label]] += 1

