#coding: utf-8
#----- 標準ライブラリ -----#
#None
#----- 専用ライブラリ -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F

#----- 自作モジュール -----#
#None



def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    precision, recall, _ = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auc_pr = metrics.auc(recall, precision)
    
    return {"fpr": fpr, "tpr": tpr, "threshold": thresholds, "auroc": auroc}




class AUROC():
    def __init__(self):
        self.__predicts = np.empty(0)
        self.__labels = np.empty(0)
    
    def _reset(self):
        self.__predicts = np.empty(0)
        self.__labels = np.empty(0)
    
    def update(self, outputs, targets):
        predicted = F.softmax(outputs, dim=-1)[:,-1]
        self.__predicts = np.append(self.__predicts, predicted.cpu().detach().numpy())
        self.__labels = np.append(self.__labels, targets.float().cpu().detach().numpy())

    def __call__(self):
        result = compute_imagewise_retrieval_metrics(self.__predicts, self.__labels)
        return result

    def keys(self):
        key_list = ["fpr", "tpr", "threshold", "auroc"]
        for i in range(len(key_list)):
            key = key_list[i]
            yield key
