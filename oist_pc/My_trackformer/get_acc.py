#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from os import path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm
from cycler import cycler as cy
from matplotlib import colors
from scipy.interpolate import interp1d


def get_mot_accum(results, seq_loader):
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for frame_id, frame_data in enumerate(seq_loader):
        gt = frame_data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, gt_box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(gt_box[0])

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack(
                (gt_boxes[:, 0],
                 gt_boxes[:, 1],
                 gt_boxes[:, 2],
                 gt_boxes[:, 3]), axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, track_data in results.items():
            if frame_id in track_data:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(track_data[frame_id]['bbox'])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                (track_boxes[:, 0],
                 track_boxes[:, 1],
                 track_boxes[:, 2] - track_boxes[:, 0],
                 track_boxes[:, 3] - track_boxes[:, 1]), axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def get_mot_accum_distance(results, seq_loader):
    mot_accum = mm.MOTAccumulator(auto_id=True)
    move_center = 6.
    right_range = 10.


    for frame_id, frame_data in enumerate(seq_loader):
        gt = frame_data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, gt_box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(gt_box[0])

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack(
                (gt_boxes[:, 0] + move_center,
                 gt_boxes[:, 1] + move_center), axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, track_data in results.items():
            if frame_id in track_data:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(track_data[frame_id]['bbox'])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                ((track_boxes[:, 0] + track_boxes[:, 2]) * 0.5,
                 (track_boxes[:, 1] + track_boxes[:, 3]) * 0.5), axis=1)
        else:
            track_boxes = np.zeros([0,2])
        
        distance = (gt_boxes[:, None] - track_boxes[None]) ** 2
        distance = np.sqrt(np.sum(distance, axis=-1))
        # 正解判定距離より大きい大きいものはnanとし、候補から外す
        distance[distance > right_range] = np.nan

        distance = distance.tolist()

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum



def evaluate_mot_accums(accums):
    calculater = mm.metrics.create()
    df = calculater.compute(accums,
                            metrics=mm.metrics.motchallenge_metrics,
                            name="Name")

    df_dict = df.to_dict(orient='list')
    # リストを解除
    for item in df_dict:
        df_dict[item] = df_dict[item][0]
    
    return df_dict
