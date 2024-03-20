#coding: utf-8
#----- 標準ライブラリ -----#
import os
import sys
import random
import copy

#----- 専用ライブラリ -----#
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#----- 自作モジュール -----#
# None

############## train関数 ##############
def train(model, train_loader, optimizer, criterion, device):
    # モデル→学習モード
    model.train()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 学習ループ  inputs:入力画像  targets:教師ラベル
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
        # GPUモード
        inputs = inputs.cuda(device,non_blocking=True)
        targets = targets.cuda(device,non_blocking=True)

        # 教師ラベルをlongモードに変換
        targets = targets.long()

        # 入力画像をモデルに入力
        output, _ = model(inputs)

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
        del loss

        ###精度の計算###

        # 出力をsoftmax関数に(0~1)
        output = F.softmax(output, dim=1)
        # 最大ベクトル
        _, predicted = output.max(1)
        # total = 正解, correct = 予測
        total += targets.size(0)
        # predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse
        correct += predicted.eq(targets).sum().item()

    return sum_loss/(batch_idx+1), correct/total


############## test関数 ##############
def test(model, test_loader, optimizer, criterion, device):
    # モデル→推論モード
    model.eval()

    # 初期設定
    sum_loss = 0
    correct = 0
    total = 0

    # 設定：パラメータの更新なし
    with torch.no_grad():
        # 学習ループ  inputs:入力画像  targets:教師ラベル画像
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):

            # GPUモード
            inputs = inputs.cuda(device,non_blocking=True)
            targets = targets.cuda(device,non_blocking=True)

            # 教師ラベルをlongモードに変換
            targets = targets.long()

            # 入力画像をモデルに入力
            output, _ = model(inputs)

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
            total += targets.size(0)
            # predicted.eq(targets) : 教師ラベルと一致していたらTrue, 不一致ならFalse
            correct += predicted.eq(targets).sum().item()

    return sum_loss/(batch_idx+1), correct/total

############## main関数 ##############
def train_test_process(model, train_loader, test_loader, out_file_path,
                       max_epoch, optimizer, lerning_rate, **kwargs):  # この行から**confing
    print(f"epoch       :{max_epoch}")
    print(f"optimizer   :{optimizer}")
    print(f"lerning rate:{lerning_rate}")

    
    #epochチェック
    if max_epoch != 100:
        print("max_epochが100ではありません")
        print("max_epochを100に変更してください")
        sys.exit(0)
    
    # device設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.cuda(device)


    # 損失関数設定
    criterion = nn.CrossEntropyLoss()  # Binaly Corss Entropy Loss

    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)

    PATH_1 = out_file_path + "/train_loss.csv"
    PATH_2 = out_file_path + "/test_loss.csv"
    PATH_3 = out_file_path + "/train_accuracy.csv"
    PATH_4 = out_file_path + "/test_accuracy.csv"

    with open(PATH_1, mode='w') as f:
        f.write("epoch\ttrain loss\n")

    with open(PATH_2, mode='w') as f:
        f.write("epoch\ttest loss\n")

    with open(PATH_3, mode='w') as f:
        f.write("epoch\ttrain accuracy\n")

    with open(PATH_4, mode='w') as f:
        f.write("epoch\ttest accuracy\n")

    # optimizer設定
    opt_dict = {"SGD": torch.optim.SGD,
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "RMSprop": torch.optim.RMSprop,
                "Adagrad": torch.optim.Adagrad,
                "Adadelta": torch.optim.Adadelta}
    
    if opt_dict.get(optimizer) is None:
        print(f"optimizer:{optimizer}は使用できません")
        for key in opt_dict.keys():
            print(f"\"{key}\"", end=',')
        print("の中から選んでください")
        sys.exit(0)
    
    optimizer = opt_dict[optimizer](model.parameters(), lr=lerning_rate)

    # 初期値の乱数設定
    random.seed(0)  # python
    np.random.seed(0)  # numpy
    torch.manual_seed(0)  # pytorch
    torch.backends.cudnn.benchmark = True

    print("##################")
    print("# training start #")
    print("##################")

    save_model = copy.deepcopy(model).to('cpu')

    result_text = "Epoch{:3d}/{:3d}  TrainLoss = {:.4f}  TrainAccuracy = {:.2%}  TestLoss = {:.4f}  TestAccuracy = {:.2%}"

    best_acc = 0.
    for epoch in range(1, max_epoch + 1):
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, device)  # train
        test_loss, test_accuracy = test(
            model, test_loader, optimizer, criterion, device)  # test

        ##### 結果表示 #####
        print(result_text.format(epoch, max_epoch, train_loss,
              train_accuracy, test_loss, test_accuracy))

        ##### 出力結果を書き込み #####
        with open(PATH_1, mode='a') as f:
            f.write(f"{epoch}\t{train_loss:.2f}\n")
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch}\t{test_loss:.2f}\n")
        with open(PATH_3, mode='a') as f:
            f.write(f"{epoch}\t{train_accuracy:.2%}\n")
        with open(PATH_4, mode='a') as f:
            f.write(f"{epoch}\t{test_accuracy:.2%}\n")

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            # save_modelにパラメータを保存
            save_model = copy.deepcopy(model).to('cpu')

    # 結果表示
    print("###################")
    print("# training finish #")
    print("###################")
    print(f"Test Accuracy   :   {best_acc:.2%}")

    return save_model
