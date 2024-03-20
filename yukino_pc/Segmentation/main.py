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
from Net import SegNet
from dataset import Drosophila_Loader
from evaluation import IoU
############################


############## dataloader関数##############
def dataload():
    # dataset = [image, label]
    # datatype = "train" or "val" or "test", transform = augmentation and preprocceing
    # val_area is for cross-validation
    train_dataset = Drosophila_Loader(rootdir="Dataset", val_area=args.split, split='train', iteration_number=12 * args.batchsize)
    val_dataset = Drosophila_Loader(rootdir="Dataset", val_area=args.split, split='val')

    # dataloader作成  batch_size=ミニバッチサイズ, shuffle(=True)=データをシャッフル,drop_last(=True)=ミニバッチサイズに合うようにデータを補う
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.Tbatchsize, shuffle=False, drop_last=False)

    return train_loader, val_loader


############## train関数 ##############
def train(epoch):
    eval_calculater = IoU(5)
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):

        # GPUモード 
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output = model(inputs)

        # 損失計算
        loss = criterion(output, targets)

        # 勾配を0に
        optimizer.zero_grad()

        # 誤差逆伝搬 
        loss.backward()

        # パラメーター更新
        optimizer.step()

        # loss溜め
        sum_loss += loss.item()


        ###精度の計算###

        # 出力をsoftmax関数に(0~1)
        output = F.softmax(output, dim=1)

        # 最大ベクトル
        _, predicted = output.max(1)

        # total = 正解, correct = 予測
        
        ###精度の計算###
        eval_calculater.update(output, targets)
        total += (targets.size(0)*targets.size(1)*targets.size(2))
        correct += predicted.eq(targets).sum().item()  #predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse
    eval_result = eval_calculater()
    print(eval_result)

    return sum_loss/(batch_idx+1), correct/total


############## validation関数 ##############
def val(epoch):
    eval_calculater = IoU(5)
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, leave=False)):
            # GPUモード  
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            #入力画像をモデルに入力
            output = model(inputs)

            # 損失計算
            loss = criterion(output, targets)

            # loss溜め
            sum_loss += loss.item()

            ###精度の計算###

            # 出力をsoftmax関数に(0~1)
            output = F.softmax(output, dim=1)
            # 最大ベクトル
            _, predicted = output.max(1)
            # total = 正解, correct = 予測
            eval_calculater.update(output, targets)
            
            total += (targets.size(0)*targets.size(1)*targets.size(2))
            correct += predicted.eq(targets).sum().item()  #predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse
    eval_result = eval_calculater()
    print(eval_result)
    return sum_loss/(batch_idx+1), correct/total


############## main ##############
if __name__ == '__main__':
    ##### コマンド設定 #####
    parser = argparse.ArgumentParser(description='SemanticSegmentation')
    # GPU番号指定
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU id')
    # データのスプリット(split)指定
    parser.add_argument('--split', '-s', type=int, default=1,
                        help='Split of data to be used')
    # ミニバッチサイズ(train)指定
    parser.add_argument('--batchsize', '-b', type=int, default=4,
                        help='Number of images in each mini-batch')
    # ミニバッチサイズ(validation)指定
    parser.add_argument('--Tbatchsize', '-t', type=int, default=4,
                        help='Number of images in each mini-batch')
    # 学習回数(epoch)指定
    parser.add_argument('--num_epochs', '-e', type=int, default=200,
                        help='Number of epoch')# 学習回数(epoch)指定
    # 出力保存先指定
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')
    # 乱数指定
    parser.add_argument('--rand_seed', '-r', type=int, default=2,
                        help='Random seed')
    # 学習率指定
    parser.add_argument('--lr', '-l', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()
    ######################

    ##### Drosophila dataset #####

    classes = 5   # class number
    inputs_ch = 1  # input channel


    ##### 初期設定表示 #####

    print("[Experimental conditions]")
    print(" GPU ID         : {}".format(args.gpu))
    print(" Epochs         : {}".format(args.num_epochs))
    print(" Minibatch size : {}".format(args.batchsize))
    print(" Class number   : {}".format(classes))
    print(" Learning rate  : {}".format(args.lr))
    print("")


    ##### GPU設定 #####

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


    ##### 保存ディレクトリ・ファイル #####

    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))

    PATH_1 = "{}/trainloss.txt".format(args.out)
    PATH_2 = "{}/valloss.txt".format(args.out)
    PATH_3 = "{}/trainaccuracy.txt".format(args.out)
    PATH_4 = "{}/valaccuracy.txt".format(args.out)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass


    # モデル設定
    model = SegNet(in_ch=inputs_ch, n_class=classes).cuda(device) #in_ch = input channel, n_class = output channel


    # 損失関数設定
    criterion = nn.CrossEntropyLoss() #Softmax Corss Entropy Loss


    # オプティマイザー設定
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#Adam
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) #SGD


    # 初期値の乱数設定
    random.seed(args.rand_seed) #python
    np.random.seed(args.rand_seed) #numpy
    torch.manual_seed(args.rand_seed) #pytorch


    # データ読み込み+初期設定 
    train_loader, val_loader = dataload()


    ##### training & validation #####
    best_loss = 1000
    #args.num_epochs = max epoch
    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = train(epoch) # train
        val_loss, val_accuracy = val(epoch) # validation

        ##### 結果表示 #####
        print("Epoch{:3d}/{:3d}  TrainLoss={:.4f}  ValAccuracy={:.2f}%".format(epoch+1,args.num_epochs,train_loss,val_accuracy*100))


        ##### 出力結果を書き込み #####
        with open(PATH_1, mode = 'a') as f:
            f.write("{}\t{:.2f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode = 'a') as f:
            f.write("{}\t{:.2f}\n".format(epoch+1, val_loss))
        with open(PATH_3, mode = 'a') as f:
            f.write("{}\t{:.2f}\n".format(epoch+1, (train_accuracy*100)))
        with open(PATH_4, mode = 'a') as f:
            f.write("{}\t{:.2f}\n".format(epoch+1, (val_accuracy)*100))

        ##### 学習済みモデル保存 #####
        # train_lossが最も低い時
        if train_loss <= best_loss:
            best_loss = train_loss
            PATH ="{}/model.pth".format(args.out)
            torch.save(model.state_dict(), PATH) #モデル保存