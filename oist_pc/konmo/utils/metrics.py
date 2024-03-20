# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


#Yolov5から引用
def ap_per_class(tp, conf, num_label, plot=True, save_dir='.', eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(conf)[::-1]
    tp, conf = tp[i], conf[i]

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((tp.shape[1])), np.zeros((1000)), np.zeros((1000))
    num_label              # number of labels
    n_p = conf.shape[0] # number of predictions
    if n_p == 0 or num_label == 0:
        return 0., 0., 0., 0., 0., np.zeros(tp.shape[-1]), 0.

    # Accumulate FPs and TPs
    tpc = tp.cumsum(0)
    fpc = (1 - tp[i]).cumsum(0)

    # Recall
    # recall = tpc / (tpc + eps)  # recall curve
    recall = tpc / (num_label + eps)  # recall curve
    r = np.interp(-px, -conf, recall[:, 0], left=0)  # negative x, xp because xp decreases

    # Precision
    precision = tpc / (tpc + fpc + eps)  # precision curve
    p = np.interp(-px, -conf, precision[:, 0], left=1)  # p at pr_score

    # AP from recall-precision curve
    for j in range(tp.shape[1]):
        ap[j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
        if plot and j == 0:
            py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png')
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', ylabel='Recall')

    i = smooth(f1, 0.1).argmax()  # max F1 index
    p, r, f1, conf = p[i], r[i], f1[i], px[i]
    tp = (r * num_label).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, conf


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec



# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png')):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    # for i, y in enumerate(py.T):
    #     ax.plot(px, y, linewidth=1, label=f'{ap[i, 0]:.3f}')  # plot(recall, precision)
    ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes mAP:10')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py, 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    tp = np.random.randint(0, 2, [100, 3])
    conf = np.random.rand(100)
    num_label = 200

    tp, fp, p, r, f1, ap, th = ap_per_class(tp, conf, num_label)
    print(p)
    print(th)
    mAP50, mAP = ap[0], ap.mean()  # AP@0.5, AP@0.5:0.95
    mp, mr = p.mean(), r.mean()
    print(mp, mr, mAP50, mAP)
    print(f1)