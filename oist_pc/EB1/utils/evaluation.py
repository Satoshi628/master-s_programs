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
from .metrics import ap_per_class

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

    def update(self, predict, label):
        """Calculation of Evaluation

        Args:
            predict (list[tensor[num,3(x,y,score)]]): predict detection coordinate.PAD=-100
            label (list[tensor[num,2(frame,id,x,y)]]): label detection coordinate.PAD=-100
        """

        if len(predict) == 0:
            for item_label in label:
                self.FN += item_label.shape[0]
        else:
            for item_pre, item_label in zip(predict, label):
                item_pre = item_pre[:,:2].float()
                item_label = item_label[:,2:].float()

                # [batch, predict, label]
                distance = torch.norm(item_pre[:, None] - item_label[None, :], dim=-1)

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

    def keys(self):
        key_list = ["Accuracy",
                    "Precition",
                    "Recall",
                    "F1_Score"]
        for k in key_list:
            yield k



class Calc_AP():
    def __init__(self, right_range=[10., 15.]):

        self.right_range = right_range
        self.TP = {str(int(r)): [] for r in right_range}
        self.conf = []
        self.num_label = 0
        #kernel_size = 4*sigma+0.5
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = 0.1

    def _reset(self):
        self.FP = 0
        self.FN = 0
        self.TP = 0

    
    def coordinater(self, outputs):
        """Function to detect the position of an object from a probability map

        Args:
            out (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """       
        position = outputs[:, 0].flatten(0, 1)

        flag = (position >= self.noise_strength) * (position == self.maxpool(position)) * position
        for _ in range(5):
            flag = self.gaussian_filter(flag)
            flag = (flag != 0) * (flag == self.maxpool(flag)) * position

        # [detection number, 3(batch*length,y,x)]
        coordinate = torch.nonzero(flag)

        scores = outputs[:, 0].flatten(0, 1)[coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]]
        # [batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        scores = scores.detach().to("cpu")

        detection = torch.cat([coordinate, scores[:, None]], dim=-1)
        detection = [detection[detection[:, 0] == batch, 1:] for batch in range(position.size(0))]

        return detection  # [batch*length] [num,3(x, y, socre)]



    def update(self, outputs, label):
        """Calculation of Evaluation

        Args:
            outputs (tensor[batch,1,length,H,W]): existly probabilty
            label (list[tensor[num,2(frame,id,x,y)]]): label detection coordinate.PAD=-100
        """
        predict = self.coordinater(outputs)

        for item_pre, item_label in zip(predict, label):
            item_pre = item_pre.float()
            item_label = item_label[:, 2:].float()
            # [batch, predict, label]
            distance = torch.norm(item_pre[:, None, :2] - item_label[None, :, :], dim=-1)

            distance = distance.to('cpu').detach().numpy().copy()
            if distance.shape[-1] == 0:
                return 
            dist_v = distance.min(axis=-1)
            dist_idx = distance.argmin(axis=-1)

            for r in self.right_range:
                TP_flag = dist_v <= r
                self.TP[str(int(r))].append(TP_flag*1.0)
            self.conf.append(item_pre[:, -1])
            self.num_label += item_label.shape[0]

    def __call__(self):
        TP = np.stack([np.concatenate(v) for v in self.TP.values()], axis=-1)
        conf = np.concatenate(self.conf)
        num_label = self.num_label
        tp, fp, p, r, f1, ap, th = ap_per_class(TP, conf, num_label)
        mAP50, mAP = ap[0], ap.mean()  # AP@0.5, AP@0.5:0.95

        if th < self.noise_strength:
            th = self.noise_strength

        self._reset()
        result = {
            "threshold": th,
            "Precition": p,
            "Recall": r,
            "F1": f1,
            f"mAP:{str(int(self.right_range[0]))}": mAP50,
            "mAP":mAP
        }
        return result

    def keys(self):
        key_list = ["threshold",
                    "Precition",
                    "Recall",
                    "F1",
                    f"mAP:{str(int(self.right_range[0]))}",
                    "mAP"]
        for k in key_list:
            yield k

class Object_Tracking():
    def __init__(self, right_range=10.):
        self.right_range = right_range
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, predict, label, video_length):
        """Add tracking results

        Args:
            predict (tensor[pre_track_num,5(frame,id,x,y,score)]): predicted track
            label (list[tensor[label_track_num,5(frame,id,x,y,score)]]): label track
        """
        #frameは0から
        
        for frame in range(video_length):
            
            pre = predict[predict[:, 0] == frame][:, [1, 2, 3]]
            lab = label[frame][:, [1, 2, 3]]

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


