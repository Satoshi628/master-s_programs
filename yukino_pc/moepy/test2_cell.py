#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import argparse
import random
from tqdm import tqdm
#from models.Unet import UNet
from models.Unet import UNet
from utils.dataset import Cell_Dataset_Loader
import utils.utils as ut
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter
import time
############################


######## IoUの測定関数 ########
def IoU(output, targets, label):
    # numpy型に変換 
    output = np.array(output)
    targets = np.array(targets)
    #print("IoU o",output.shape)#(2, 4, 4, 256, 256)
    #print("IoU t",target.shape)#(2, 4, 256, 256)

    # 確率最大値のラベルを出力
    seg = np.argmax(output,axis=2)
    #print("IoU seg1",seg.shape)#(2, 4, 256, 256)

    # 1次元に
    seg = seg.flatten()
    targets = targets.flatten()
    #print("IoU seg2",seg.shape)#(524288)
    #print("IoU tage",target.shape)#(524288)

    # seg = ネットワークの予測
    # target = 正解ラベル 
    mat = confusion_matrix(targets, seg, labels=label)
    #print("IoU mat",mat.shape)#(4,4)

    # IoU計算
    iou_den = (mat.sum(axis=1) + mat.sum(axis=0) - np.diag(mat))#axis=1横方向, axis=0縦方向, np.diag()対角成分抽出
    iou = np.array(np.diag(mat) ,dtype=np.float32) / np.array(iou_den, dtype=np.float32)

    return iou


############## dataloader 関数##############
def dataload():
    # data preprocceing

    test_transform = ut.Compose([ut.ToTensor(), #0~1正規化+Tensor型に変換
                                 ut.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #データの標準化
                                 ])

    # dataset = [image, label]
    # datatype = "train" or "val" or "test", transform = augmentation and preprocceing

    #test_dataset = Covid19_Loader(dataset_type='test', transform=test_transform)
    test_dataset = Cell_Dataset_Loader(dataset_type='test', transform=test_transform)
    # class数変えるの忘れないように

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.Tbatchsize, shuffle=False, drop_last=True)


    return test_loader



############## test関数 ##############
def test():
    # モデル→推論モード
    model.eval()

    # 初期設定
    correct = 0
    total = 0
    c = 0
    num = 0

    predict = []
    answer = []
    labels = np.arange(3)

    start_time = time.time()

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            # GPUモード  
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            #入力画像をモデルに入力
            #y, output = model(inputs)
            #y = model(inputs)
            #y, h_0, h_1, h_2, h_3 = model(inputs)
            y, h_0, h_1, h_2 = model(inputs)

            ###精度の計算###

            # 出力をsoftmax関数に(0~1)
            output = F.softmax(y, dim=1)
            # 最大ベクトル
            _, predicted = output.max(1)

            # 画像保存
            for i in range(predicted.size(0)):
                num += 1
                item = predicted[i]
                img = np.zeros((256, 256, 3))
                for k in range(256):
                    for l in range(256):
                        if (item[k][l] == 0):
                            img[k][l] = (0, 0, 0)
                        elif (item[k][l] == 1):
                            img[k][l] = (255, 0, 0)
                        elif (item[k][l] == 2):
                            img[k][l] = (0, 0, 255)
                img = Image.fromarray(np.uint8(img))
                
                # img数字.pngで色分けして保存
                #name = 'iimg' + str(num) + '.png'
                name = args.out + "/"+ 'img' + str(num) + '.png'
                img.save(name)
            # total = 正解, correct = 予測 
            total += (targets.size(0)*targets.size(1)*targets.size(2))
            correct += predicted.eq(targets).sum().item()  #predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse

            # 評価用にTensor型→numpy型に変換
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()
            #print("test out2",output.shape)#(16, 12, 360, 480)
            #print("test t3",targets.shape)#(16, 360, 480)
            
            # 出力溜め
            predict.append(output)
            answer.append(targets)
            #print("test pred",len(predict))
            #print("test ans",len(answer))
            c += 1
            #print()

        # IoU測定
        iou = IoU(predict, answer, label=labels)
        print("iou",iou)
        
        # meanIoU
        miou = np.mean(iou)

    end_time = time.time()
    print("end_time", end_time - start_time)


    return correct/total, miou

    


############## main ##############
if __name__ == '__main__':
    ##### コマンド設定 #####
    parser = argparse.ArgumentParser(description='SemanticSegmentation')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU id')
    # ミニバッチサイズ(teste)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4,
                        help='Number of images in each mini-batch')
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')

    args = parser.parse_args()
    ######################



    ##### Covid-19 dataset #####

    classes = 3   # class number
    inputs_ch = 3  # input channel


    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ##### 保存ディレクトリ・ファイル #####

    PATH = "{}/predict.txt".format(args.out)

    with open(PATH, mode = 'w') as f:
        pass


    # モデル設定
    model = UNet(3,3).cuda(device)


    # データ読み込み+初期設定 
    test_loader = dataload()


    # 学習済みモデルのロード
    #model_path = PATH ="{}/model.pth".format(args.out)
    #model.load_state_dict(torch.load(model_path))

    model_path = PATH ="result217/model.pth"
    model.load_state_dict(torch.load(model_path))

    ##### test #####
    test_accuracy, miou = test()


    ##### 結果表示 #####
    #print("TestAccuracy==={:.2f}%".format(test_accuracy*100))
    print("TestAccuracy==={:.2f}%  miou={:.2f}%".format(test_accuracy*100,miou*100))

    ##### 出力結果を書き込み #####
    with open(PATH, mode = 'a') as f:
        f.write("{:.2f}\n".format(test_accuracy))


