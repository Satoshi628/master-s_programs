#coding: utf-8
#----- Standard Library -----#
import os
from functools import lru_cache
import time

#----- Public Package -----#
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import motmetrics as mm

#----- Module -----#
# None


PAD_ID = -100

class Object_Detection():
    def __init__(self, right_range=10.,
                noise_strength=0.5,
                pool_range=11):
        self.FP = 0
        self.FN = 0
        self.TP = 0

        self.right_range = right_range

        self.coordinater_maximul
        #kernel_size = 4*sigma+0.5
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(pool_range, stride=1, padding=(pool_range - 1) // 2)
        self.noise_strength = noise_strength

    def coordinater_maximul(self, out):
        """ Function to return the detected position of an object from a probability map.

        Args:
            out (tensor[batch,1,H,W]): The probability map section is [0,1].

        Returns:
            tensor[batch,2(x,y)]: detection coordinate.
        """
        position = out[:, 0].flatten(0, 1)

        position = (position >= self.noise_strength) * (position == self.maxpool(position)) * 1.0
        for _ in range(5):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position == self.maxpool(position)) * 1.0

        # [detection number, 3(batch*length,y,x)]
        coordinate = torch.nonzero(position)
        # [batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(position.size(0))]
        return coordinate  # [num_cell,3] 1dim=[batch_num,y,x]

    def _reset(self):
        self.FP = 0
        self.FN = 0
        self.TP = 0

    def calculate(self, predict, label):
        """Calculation of Evaluation

        Args:
            predict (tensor[batch,num,2(x,y)]): predict detection coordinate.PAD=-100
            label (tensor[batch,num,2(x,y)]): label detection coordinate.PAD=-100
        """

        predict = predict
        label = label

        if len(predict) == 0:
            for item_label in label:
                item_label = item_label[item_label[:, 0] != PAD_ID]
                self.FN += item_label.shape[0]
        else:
            for item_pre, item_label in zip(predict, label):
                item_pre = item_pre[item_pre[:, 0] != PAD_ID].float()
                item_label = item_label[item_label[:, 0] != PAD_ID].float()

                # [batch, predict, label]
                distance = torch.norm(item_pre[:, None, :] - item_label[None, :, :], dim=-1)

                distance = distance.to('cpu').detach().numpy().copy()

                row, col = linear_sum_assignment(distance)

                correct_flag = distance[row, col] <= self.right_range

                TP = correct_flag.sum().item()
                self.TP += TP
                self.FP += item_pre.shape[0] - TP
                self.FN += item_label.shape[0] - TP

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
        method = {member for member in vars(
            self.__class__) if not "__" in member}

        return method

    def __dir__(self):
        method = [member for member in dir(
            self.__class__) if not "__" in member]

        return method


class Object_Tracking():
    def __init__(self, right_range=10.):
        self.right_range = right_range
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, predict, label):
        """Add tracking results

        Args:
            predict (tensor[pre_track_num,4(frame,id,x,y)]): predicted track
            label (tensor[label_track_num,4(frame,id,x,y)]): label track
        """
        
        pre_min_f = predict[:, 0].min()
        pre_max_f = predict[:, 0].max()
        label_min_f = label[:, 0].min()
        label_max_f = label[:, 0].max()

        predict[:, 0] = predict[:, 0] - pre_min_f
        label[:, 0] = label[:, 0] - label_min_f

        """
        Prediction results use the number of frames in the label 
        because the video is longer depending on the data loader.
        """
        label_max_f = label_max_f - label_min_f
        for frame in range(int(label_max_f.item())):
            
            pre = predict[predict[:, 0] == frame][:, [1, 2, 3]]
            lab = label[label[:, 0] == frame][:, [1, 2, 3]]

            # delete PAD
            pre = pre[pre[:, 1] >= 0].cpu().detach().float()
            lab = lab[lab[:, 1] >= 0].cpu().detach().float()
            if pre is None:
                continue

            dist_matrix = self.distance_calculate(pre, lab)

            pre_ids = pre[:, 0].long().tolist()
            lab_ids = lab[:, 0].long().tolist()

            self.acc.update(lab_ids, pre_ids, dist_matrix)

    def distance_calculate(self, predict, label):
        # predict,label size => [pre_num,3(id,x,y)],[label_num,3(id,x,y)]

        # dist_matrix.size() => [label_num,pre_num]
        dist_matrix = torch.norm(label[:, None, [1, 2]] - predict[None, :, [1, 2]], dim=-1)

        dist_matrix[dist_matrix > self.right_range] = np.nan

        return dist_matrix.tolist()

    def _reset(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def __call__(self):

        calculater = mm.metrics.create()
        metrics = mm.metrics.motchallenge_metrics + ['num_objects']

        df = calculater.compute(self.acc,
                                metrics=metrics,
                                name="Name")
        print(mm.io.render_summary(df, formatters=calculater.formatters, namemap=mm.io.motchallenge_metric_names))

        df_dict = df.to_dict(orient='list')
        
        for item in df_dict:
            df_dict[item] = df_dict[item][0]
        self._reset()
        return df_dict
