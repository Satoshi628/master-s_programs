#coding: utf-8
#----- 標準ライブラリ -----#
import os
from functools import lru_cache
import time
#----- 専用ライブラリ -----#
from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import motmetrics as mm
import h5py
#----- 自作ライブラリ -----#
# None


class Object_Detection():
    def __init__(self, right_range=10.,
                mode="maximul",
                noise_strength=0.5,
                pool_range=11):
        self.FP = 0
        self.FN = 0
        self.TP = 0

        self.right_range = right_range

        mode_list = ["maximul"]
        self.mode = mode

        if not mode in mode_list:
            raise KeyError("modeは{}のどれかでなければなりません。mode={}".format(mode_list, mode))

        if mode == mode_list[0]:  # maximul
            self.coordinater = self.coordinater_maximul
            #ガウシアンカーネル定義:kernel_size = 4*sigma+0.5
            self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
            # 極大値を探索するためのMaxpooling
            self.maxpool = nn.MaxPool2d(pool_range, stride=1, padding=(pool_range - 1) // 2)
            # ノイズの閾値を決める変数。これ以下の確信度のものはノイズとして検出する
            self.noise_strength = noise_strength

    def coordinater_maximul(self, out):
        """確率マップから物体の検出位置を返す関数。極大から検出位置を計算する。

        Args:
            out (tensor[batch,1,H,W]): 確率マップ区間は[0,1]

        Returns:
            tensor[batch,2(x,y)]: 検出位置
        """
        h = out.squeeze(1)
        h = (h >= self.noise_strength) * (h == self.maxpool(h)) * 1.0
        for _ in range(3):
            h = self.gaussian_filter(h)
            h = (h != 0) * (h == self.maxpool(h)) * 1.0

        coordinate = torch.nonzero(h)  # [detection number, 3(batch,y,x)]となる。
        if coordinate.size(0) == 0:
            return coordinate
        coordinate = self.coordinate_batch_convert(coordinate, out.size(0))
        return coordinate  # [num_cell,3] 1dim=[batch_num,y,x]

    def coordinate_batch_convert(self, coord, batch_size):
        # 位置情報をbatchごとに分ける。[batch, max detection number, 2(x,y)]となるようにする
        #cell_max_count * frameの配列に入れるための番号を作製
        # batchに一つも値がない場合のケア
        batch_num = torch.cat([coord[:, 0], torch.arange(
            batch_size, device=coord.device)], dim=0)

        _, count = torch.unique(batch_num, sorted=True, return_counts=True)
        count -= 1

        max_count = torch.max(count)
        batch_idx = coord.new_ones(coord.shape[0])
        batch_idx[0] = 0

        # batch_idx[torch.cumsum(count[:-1], dim=0)] +=  max_count - count[:-1]では同じところを複数加算できない
        # 仕方ないのでfor文,batch sizeは小さいため許容
        for idx, value in zip(torch.cumsum(count[:-1], dim=0), max_count - count[:-1]):
            try:
                batch_idx[idx] += value
            except IndexError:  # 最後のbatchのデータ数が0だとエラーになる。
                pass
        batch_idx = torch.cumsum(batch_idx, dim=0)

        batch_coord = -torch.ones([batch_size * max_count, 2],dtype=torch.long, device=coord.device)
        batch_coord[batch_idx] = coord[:, [2, 1]]  # [y,x] => [x,y]
        batch_coord = batch_coord.view(batch_size, max_count, 2)
        return batch_coord

    def _reset(self):
        self.FP = 0
        self.FN = 0
        self.TP = 0

    def calculate(self, predict, label):
        """評価指標計算。TP,FP,FNをため込む関数。for文ないで使う

        Args:
            predict (tensor[batch,num,2(x,y)]): 予測検出位置。PADには-1が入る
            label (tensor[batch,num,2(x,y)]): 正解検出位置。PADには-1が入る
        """

        predict = predict.float()
        label = label.float()

        # もし一つも検出されなかったら
        if predict.size(0) == 0:
            for item_label in label:
                item_label = item_label[item_label[:, 0] != -1]
                self.FN += item_label.size(0)
        else:
            # 割り当て問題、どのペアを選べば距離が最も小さくなるか
            for item_pre, item_label in zip(predict, label):
                item_pre = item_pre[item_pre[:, 0] != -1]
                item_label = item_label[item_label[:, 0] != -1]

                # distanceは[batch,predict,label]となる
                distance = torch.norm(
                    item_pre[:, None, :] - item_label[None, :, :], dim=-1)

                distance = distance.to('cpu').detach().numpy().copy()

                row, col = linear_sum_assignment(distance)

                # 距離がマッチング閾値より低いかを判別
                correct_flag = distance[row, col] <= self.right_range

                # TP,FN,FPを計算
                TP = correct_flag.sum().item()

                self.TP += TP
                self.FP += item_pre.size(0) - TP
                self.FN += item_label.size(0) - TP

    def __call__(self):
        Accuracy = self.TP / (self.TP + self.FP + self.FN + 1e-7)
        Precition = self.TP / (self.TP + self.FP + 1e-7)
        Recall = self.TP / (self.TP + self.FN + 1e-7)
        F1_Score = 2 * Precition * Recall / (Precition + Recall + 1e-7)

        self._reset()
        result = {
            "Accuracy": Accuracy,
            "Precition": Precition,
            "Recall": Recall,
            "F1_Score": F1_Score
        }
        return result

    def __vars__(self):
        # 自身が定義したメソッド、変数のみ抽出
        method = {member for member in vars(
            self.__class__) if not "__" in member}

        return method

    def __dir__(self):
        # 自身が定義したメソッド、変数のみ抽出
        method = [member for member in dir(
            self.__class__) if not "__" in member]

        return method



class Object_Tracking():
    def __init__(self, right_range=10.):
        self.right_range = right_range
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, predict, label):
        """tracking状況を更新するプログラム

        Args:
            predict (tensor[pre_track_num,4(frame,id,x,y)]): 予測tracking結果
            label (tensor[label_track_num,4(frame,id,x,y)]): ラベルtracking結果
        """
        # frameを合わせる
        pre_min_f = predict[:, 0].min()
        pre_max_f = predict[:, 0].max()
        label_min_f = label[:, 0].min()
        label_max_f = label[:, 0].max()

        #assert (label_max_f - label_min_f) == (pre_max_f - pre_min_f), "予測とラベルでtracking長さが違います"

        predict[:, 0] = predict[:, 0] - pre_min_f
        label[:, 0] = label[:, 0] - label_min_f

        # 最大値の値を修正
        pre_max_f = pre_max_f - pre_min_f
        for frame in range(int(pre_max_f.item())):
            #frameを指定し、id,x,yのみを抽出
            pre = predict[predict[:, 0] == frame][:, [1, 2, 3]]
            lab = label[label[:, 0] == frame][:, [1, 2, 3]]

            # PAD除去
            pre = pre[pre[:, 1] >= 0].cpu().detach().float()
            lab = lab[lab[:, 1] >= 0].cpu().detach().float()
            #print(pre)
            #print(lab)
            #input()
            if pre is None:
                continue

            dist_matrix = self.distance_calculate(pre, lab)

            pre_ids = pre[:, 0].long().tolist()
            lab_ids = lab[:, 0].long().tolist()

            self.acc.update(lab_ids, pre_ids, dist_matrix)

    def distance_calculate(self, predict, label):
        # predict,labelの大きさはそれぞれ[pre_num,3(id,x,y)],[label_num,3(id,x,y)]
        # PADは含まれないものとする

        # dist_matrixの大きさは[label_num,pre_num]
        dist_matrix = torch.norm(label[:, None, [1, 2]] - predict[None, :, [1, 2]], dim=-1)

        # 正解判定距離より大きい大きいものはnanとし、候補から外す
        dist_matrix[dist_matrix > self.right_range] = np.nan

        return dist_matrix.tolist()

    def _reset(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def __call__(self):

        calculater = mm.metrics.create()

        df = calculater.compute(self.acc,
                                metrics=mm.metrics.motchallenge_metrics,
                                name="Name")

        df_dict = df.to_dict(orient='list')
        # リストを解除
        for item in df_dict:
            df_dict[item] = df_dict[item][0]
        self._reset()
        return df_dict
