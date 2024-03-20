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
#None

def MOTA(predict_cell_track, label_cell_track, range=10.):
    """
    参考資料:https://www.ariseanalytics.com/activities/report/20201016/
    False Positive
    Misses
    ID Switch
    の3つを作製しなければならないID Switchの作り方が分からないのでパス
    """
    pass



class Precition_Recall():
    def __init__(self, right_range=10.):
        """精度計算用Class

        Args:
            right_range (float, optional): 正解にする距離. Defaults to 10..
        """        
        self.right_range = right_range


    def F1_calculate(self, predict_cell_track, label_cell_track):
        """Precition,Recall,F1 Scoreを計算する関数

        Args:
            predict_cell_track (torch.tensor[:,5]): モデルの追跡結果。1次元目は[frame,id,x,y,parent id]
            label_cell_track (torch.tensor[:,5]): 正解ラベルの追跡。1次元目は[frame,id,x,y,parent id]

        Returns:
            Precition (float): モデル出力のPrecition精度
            Recall (float): モデル出力のRecall精度
            F1 (float): モデル出力のF1精度
        """        
        
        pre_len = predict_cell_track[:, 0].max() - predict_cell_track[:, 0].min() + 1
        lab_len = label_cell_track[:, 0].max() - label_cell_track[:, 0].min() + 1
        #assert pre_len == lab_len, "predictとlabelのフレーム数が違います:predint={} label={}".format(pre_len, lab_len)
        label_cell_track[:, 0] -= label_cell_track[:, 0].min()
        
        # FP.size() => [predict,label],FN.size() => [label,predict]
        print("FPFN開始")
        FP = self.FP_FN_calculate(predicts=predict_cell_track, label=label_cell_track, mode="FP")
        FN = self.FP_FN_calculate(predicts=predict_cell_track, label=label_cell_track, mode="FN")
        
        # 長さ調節
        _,FP_pad_value = torch.unique(predict_cell_track[:, 1], return_counts=True)
        _, FN_pad_value = torch.unique(label_cell_track[:, 1], return_counts=True)
        

        pre_len = FP_pad_value.size(0)
        lab_len = FN_pad_value.size(0)
        diff_len = pre_len - lab_len
        if diff_len > 0:  # もしpredictの方がidが多かったら
            # labelを増やす
            FP_pad_value = FP_pad_value.unsqueeze(-1)
            FP_pad_value = FP_pad_value.expand(-1, diff_len)
            # 長さ0のtrack結果を増やす
            FN_pad_value = torch.zeros((diff_len, pre_len))
            FP = torch.cat([FP, FP_pad_value], dim=1)
            FN = torch.cat([FN, FN_pad_value], dim=0)
        elif diff_len < 0:  # もしlabelの方がidが多かったら
            # predictを増やす
            # 長さ0のtrack結果を増やす
            FP_pad_value = torch.zeros((-diff_len, lab_len))
            FN_pad_value = FN_pad_value.unsqueeze(-1)
            FN_pad_value = FN_pad_value.expand(-1, -diff_len)
            FP = torch.cat([FP, FP_pad_value], dim=0)
            FN = torch.cat([FN, FN_pad_value], dim=1)
        
        #FNとTPの次元の意味を同じにする
        FN = FN.t()

        # 最小の物を抽出しIDFP,IDFNにする
        print("開始")
        _, min_idx = self.Kuhn_Munkres_algorithm(FP + FN)
        
        IDFP = FP[min_idx[:, 0], min_idx[:, 1]].sum()
        IDFN = FN[min_idx[:, 0], min_idx[:, 1]].sum()
        assert predict_cell_track.size(0) - IDFP == label_cell_track.size(0) - IDFN, "計算にミスがあります。IDTP no match"
        IDTP = predict_cell_track.size(0) - IDFP

        # 精度計算
        precition = IDTP / (IDTP + IDFP)    #予測した結果が正しい割合
        recall = IDTP / (IDTP + IDFN)       #ラベルが予測された割合
        F1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        return precition.item(), recall.item(), F1.item()
    
    def FP_FN_calculate(self, predicts, label, mode="FP"):
        """FPとFNを計算するための関数。modeで切り替え可能

        Args:
            predicts (torch.tensor): 細胞の追跡結果
            label (torch.tensor): 細胞追跡の正解ラベル
            mode (str, optional): モードを切り替えれる。FNのときは"FN"と記載. Defaults to "FP".

        Returns:
            torch.tensor[predict_cell_id,label_cell_id]: 計算結果、FP,FNの数を示す
        """        
        if mode == "FN":
            temp = predicts
            predicts = label
            label = temp
        predicts_ids = torch.unique(predicts[:, 1])
        label_ids = torch.unique(label[:, 1])

        FP_FN = torch.zeros((predicts_ids.size(0), label_ids.size(0)))

        for pre_idx, predicts_id in enumerate(predicts_ids):
            for lab_idx, label_id in enumerate(label_ids):
                predict_cell = predicts[predicts[:, 1] == predicts_id]
                label_cell = label[label[:, 1] == label_id]
                # TP_count.size()=>[predict_cell frame,label_cell frame]
                TP_count = predict_cell[:,0].view(-1, 1) == label_cell[:, 0].view(1, -1)

                count = predict_cell.size(0) - TP_count.sum()

                distance = torch.norm(predict_cell[TP_count.any(1)][:, [2, 3]]
                                    - label_cell[TP_count.any(0)][:, [2, 3]], dim=-1)

                count += (distance > self.right_range).sum()
                FP_FN[pre_idx, lab_idx] = count

        return FP_FN

    def Kuhn_Munkres_algorithm(self, matrix):
        assert matrix.dim() == 2, "2次元配列ではありません"
        assert matrix.size(0) == matrix.size(1), "正方行列の形が合っていません"
        temp_matrix = matrix.clone()  # 入力を変化させないようにクローンを作製

        #step1:行の最小値を行ごとに引き、列でも同じことをする。
        temp_matrix = temp_matrix - temp_matrix.min(dim=1)[0].unsqueeze(1)
        temp_matrix = temp_matrix - temp_matrix.min(dim=0)[0].unsqueeze(0)

        while True:
            #step2:各行、列すべてに0が含まれているか確認する。Trueなら探索終了。Falseならstep3へ
            flag, idx = self.choice_zero(temp_matrix)
            if flag:
                idx = idx.long()
                minimum = matrix[idx[:, 0], idx[:, 1]].sum()

                return minimum, idx.long()

            #step3:全てのゼロをできるだけ少ない線で被せる
            rows, cols = self.zero_line_algorithm(temp_matrix, idx.long())
            #line_matrix, rows, cols = zero_line(temp_matrix)

            #step4:線引きした行列から最小値をとり、元の行列に差し引く。そしてstep2に戻る
            #線を引かない箇所
            non_liner = (rows.view(-1, 1) & cols.view(1, -1))
            #線が交差する場所
            cross_liner = (~rows.view(-1, 1) & ~cols.view(1, -1))
            #print(non_liner)
            #print(cross_liner)
            line_min = temp_matrix[non_liner].min()

            #print(temp_matrix)
            temp_matrix[non_liner] -= line_min
            temp_matrix[cross_liner] += line_min
            #print(temp_matrix)
            #input()


    def choice_zero(self, matrix):
        """0を含む2次元配列からすべての行、列から1つ、ゼロを選べるかどうか判定する

        Args:
            matrix (torch.tensor): 2次元配列

        Returns:
            bool: 選択できるかどうか
            torch.tensor: 選択できたindex
        """
        zero_matrix = matrix == 0
        result_index = matrix.new_empty((0, 2))

        for i in reversed(range(1, zero_matrix.size(0)+1)):
            #行のゼロカウント
            zero_count_row = zero_matrix.sum(dim=0).to(float)
            #列のゼロカウント
            zero_count_col = zero_matrix.sum(dim=1).to(float)

            zero_count_row[zero_count_row == 0] = float('Inf')
            zero_count_col[zero_count_col == 0] = float('Inf')

            #行のカウントの最小値とidx取得
            row_count_min, row_count_idx = zero_count_row.min(dim=0)
            #列のカウントの最小値とidx取得
            col_count_min, col_count_idx = zero_count_col.min(dim=0)

            if torch.isinf(row_count_min):
                return False, result_index
            if torch.isinf(col_count_min):
                return False, result_index
            #行の判定
            if row_count_min == 1:
                temp_idx = torch.nonzero(zero_matrix[:, row_count_idx])
                temp_idx = matrix.new_tensor([[temp_idx, row_count_idx]]).long()
                result_index = torch.cat([result_index, temp_idx], dim=0)
                zero_matrix[temp_idx[0, 0]] = False
                zero_matrix[:, row_count_idx] = False
            #列の判定
            elif col_count_min == 1:
                temp_idx = torch.nonzero(zero_matrix[col_count_idx, :])
                temp_idx = matrix.new_tensor([[col_count_idx, temp_idx]]).long()
                result_index = torch.cat([result_index, temp_idx], dim=0)
                zero_matrix[col_count_idx] = False
                zero_matrix[:, temp_idx[0, 1]] = False
            else:
                temp_idx = torch.nonzero(zero_matrix)[:1]
                result_index = torch.cat([result_index, temp_idx], dim=0)
                zero_matrix[temp_idx[0, 0]] = False
                zero_matrix[:, temp_idx[0, 1]] = False

        return True, result_index


    def zero_line_algorithm(self, matrix, assign_zeros_idx):
        #アルゴリズムはこちらを参照:https://en.wikipedia.org/wiki/Hungarian_algorithm
        #マーク変数
        row_mark = matrix.new_zeros(matrix.size(0), dtype=bool)  # 行
        col_mark = matrix.new_zeros(matrix.size(1), dtype=bool)  # 列

        #行から先にマークする
        assign_zeros = matrix.new_zeros(matrix.shape, dtype=bool)
        assign_zeros[assign_zeros_idx[:, 0], assign_zeros_idx[:, 1]] = True

        row_mark = ~torch.any(assign_zeros, dim=1)  # 割り当てゼロがない行にマークをする

        while True:
            #過去のマーク状況を保存
            past_row_mark = row_mark
            past_col_mark = col_mark

            col_mark = torch.any(matrix[row_mark] == 0, dim=0)  # ゼロがある列をマークする

            #その列にassign_zerosがある行をマークする
            row_assign = torch.any(assign_zeros[:, col_mark], dim=1)

            #行のマークを更新
            row_mark = row_mark | row_assign

            #過去のマーク状況と照らし合わせ、更新がなかったら終了する
            if torch.all((past_row_mark == row_mark) & (past_col_mark == col_mark)):
                return row_mark, ~col_mark  # Falseが線を引くところ



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
        label_max_f = label_max_f - label_min_f
        for frame in range(int(label_max_f.item())):
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
        dist_matrix = torch.norm(
            label[:, None, [1, 2]] - predict[None, :, [1, 2]], dim=-1)

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


class Object_Detection():
    def __init__(self, right_range=10.):
        self.FP = 0
        self.FN = 0
        self.TP = 0
        self.right_range = right_range

    def _reset(self):
        self.FP = 0
        self.FN = 0
        self.TP = 0

    def update(self, predict, label):
        """評価指標計算。TP,FP,FNをため込む関数。for文ないで使う

        Args:
            predict (tensor[num,3(frame,x,y)]): 予測検出位置。PADには-1が入る
            label (tensor[num,3(frame,x,y)]): 正解検出位置。PADには-1が入る
        """

        predict = predict.float()
        label = label.float()
        
        #frameに区切る
        predict = [predict[predict[:, 0] == frame] for frame in torch.unique(predict[:, 0])]
        label = [label[label[:, 0] == frame] for frame in torch.unique(label[:, 0])]
        

        # 割り当て問題、どのペアを選べば距離が最も小さくなるか
        for item_pre, item_label in zip(predict, label):
            item_pre = item_pre[item_pre[:, 1] != -1]
            item_label = item_label[item_label[:, 1] != -1]

            # distanceは[batch,predict,label]となる
            distance = (item_pre[:, None, [1, 2]] - item_label[None, :, [1, 2]]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

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
        return Accuracy, Precition, Recall, F1_Score

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
