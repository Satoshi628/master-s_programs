#coding: utf-8
#----- Standard Library -----#
import os
import re
import time
import math

#----- Public Package -----#
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pulp import *
import lap

#----- Module -----#
# None

PAD_ID = -100


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.numel() == 0:
        return torch.zeros(cost_matrix.shape[0], device=cost_matrix.device, dtype=bool), -1 * torch.ones(cost_matrix.shape[0], device=cost_matrix.device, dtype=torch.long)

    cost = cost_matrix.detach().to("cpu").numpy()
    cost, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)

    x = torch.from_numpy(x)
    x = x.long().to(cost_matrix.device)
    match_flag = x >= 0

    return match_flag, x


def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """ Function to convert index tensor to one hot vector

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): Number of dimensions of one hot vector
        dim (int, optional): Dimensions of one hot vector. Defaults to -1.
        bool_ (bool, optional): If True, it will be of type bool; if False, it will be of type torch.float. Defaults to False.

    Raises:
        TypeError: index tensor is not torch.long
        ValueError: index tensor is greater than vector_dim

    Returns:
        torch.tensor: one hot vector
    """
    if bool_:
        data_type = bool
    else:
        data_type = torch.float
    if tensor.dtype != torch.long:
        raise TypeError("Input tensor is not of type torch.long.")
    if tensor.max() >= vector_dim:
        raise ValueError(
            "The index number of the input tensor is larger than vector_dim.")

    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor, ]

    dim_change_list = list(range(tensor.dim()))
    if dim == -1:
        return vector
    if dim < 0:
        dim += 1

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector

# 一対一マッチング
class MPM_to_Track_normal:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur(
            (13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = move_speed
        self.track = []
        self.max_track_id = 0

    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                dist = (coord[:, None] - self.track[-1][None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                match_flag1, dist_idx = linear_assignment(dist, self.move_speed[0])

                # マッチ処理
                matched_id = self.track[-1][dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)

                # 現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    # マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    # マッチしていない過去の検出
                    uni_ids1 = torch.unique(self.track[-1][:, 0])
                    uni_ids2 = torch.unique(self.track[-2][:, 0])
                    unmatch_track_ids = uni_ids2[~torch.any(uni_ids2[:, None] == uni_ids1[None], dim=-1)]
                    unmatch_track = self.track[-2][torch.any(self.track[-2][:, 0, None] == unmatch_track_ids[None], dim=-1)]

                    dist = (no_match_coord[:, None] - unmatch_track[:, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    match_flag2, dist_idx = linear_assignment(dist, self.move_speed[1])

                    # マッチ処理
                    matched_id = unmatch_track[dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    # 線形補間
                    matched_xy = unmatch_track[dist_idx[match_flag2], 1:]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    # マッチング処理
                    self.track[-1] = torch.cat([self.track[-1], frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])

                # マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(torch.cat([matched_track, matched2_track, new_track], dim=0))

            else:  # trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = track_id[-1]

    def MPM2vector(self, outputs):
        """Function to detect the position of an object from a probability map

        Args:
            outputs (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """
        position = outputs[:, 0].flatten(0, 1)

        position = (position >= self.noise_strength) * (position == self.maxpool(position)) * 1.0
        for _ in range(5):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position == self.maxpool(position)) * 1.0

        # [detection number, 3(batch*length,y,x)]
        coordinate = torch.nonzero(position)
        # [batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(position.size(0))]

        return coordinate  # [batch*length] [num,2(x, y)]

    def get_track(self):
        return self.track

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, length, delay=1):
        track = []
        for idx in range(len(self.track)):
            if idx % delay == 0:
                frame = idx // delay * torch.ones(self.track[idx].shape[0])
                track.append(torch.cat([frame[:, None], self.track[idx]], dim=-1))
        print(len(track))
        track = torch.cat(track[:length], dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > 2]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])

        
        uni_ids = torch.unique(track[:, 1])
        for idx, uid in enumerate(uni_ids):
            track[track[:, 1] == uid, 1] = idx
        
        print(track)
        print(track.shape)
        return track


# 一対一マッチング
class MPM_to_Track_multi:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur(
            (13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = move_speed
        self.track = []
        self.max_track_id = 0

    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                dist = (coord[:, None] - self.track[-1][None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                match_flag1, dist_idx = linear_assignment(dist, self.move_speed[0])

                # マッチ処理
                matched_id = self.track[-1][dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)

                # 現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    # マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    # マッチしていない過去の検出
                    uni_ids1 = torch.unique(self.track[-1][:, 0])
                    uni_ids2 = torch.unique(self.track[-2][:, 0])
                    unmatch_track_ids = uni_ids2[~torch.any(uni_ids2[:, None] == uni_ids1[None], dim=-1)]
                    unmatch_track = self.track[-2][torch.any(self.track[-2][:, 0, None] == unmatch_track_ids[None], dim=-1)]

                    dist = (no_match_coord[:, None] - unmatch_track[:, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    match_flag2, dist_idx = linear_assignment(dist, self.move_speed[1])

                    # マッチ処理
                    matched_id = unmatch_track[dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    # 線形補間
                    matched_xy = unmatch_track[dist_idx[match_flag2], 1:]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    # マッチング処理
                    self.track[-1] = torch.cat([self.track[-1], frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])

                # マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(torch.cat([matched_track, matched2_track, new_track], dim=0))

            else:  # trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = track_id[-1]

    def MPM2vector(self, outputs):
        """Function to detect the position of an object from a probability map

        Args:
            outputs (tensor[batch,2,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """
        position = outputs.permute(0, 2, 1, 3, 4).flatten(0, 1)

        position = (position >= self.noise_strength) * (position == self.maxpool(position)) * 1.0
        for _ in range(5):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position == self.maxpool(position)) * 1.0

        # [detection number, 3(batch*length,y,x)]
        coordinate = torch.nonzero(position)
        # [batch*length,num,3(batch*length,class,y,x)] => [batch*length,num,3(batch*length,class,x,y)]
        coordinate = coordinate[:, [0, 1, 3, 2]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(position.size(0))]

        coordinate_list = []

        for coords in coordinate:
            multi_coords = coords[coords[:, 0] == 1]
            frame_coords = [coords[coords[:, 0] == 0]]

            for m_coord in multi_coords:
                distance = (m_coord[None, 1:] - coords[:, 1:]) ** 2
                distance = np.sqrt(distance.sum(dim=-1))
                multiple_coord = coords[distance < 11.]
                if multiple_coord.shape[0] == 1:
                    frame_coords.append(multiple_coord)
            coordinate_list.append(torch.cat(frame_coords, dim=0)[:, 1:])
        
        return coordinate_list  # [batch*length] [num,2(x, y)]

    def get_track(self):
        return self.track

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, length, delay=1):
        track = []
        for idx in range(len(self.track)):
            if idx % delay == 0:
                frame = idx//delay * torch.ones(self.track[idx].shape[0])
                track.append(
                    torch.cat([frame[:, None], self.track[idx]], dim=-1))
        print(len(track))
        track = torch.cat(track[:length], dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > 2]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])

        uni_ids = torch.unique(track[:, 1])
        for idx, uid in enumerate(uni_ids):
            track[track[:, 1] == uid, 1] = idx
        
        print(track)
        print(track.shape)
        return track


# 検出追加
class MPM_to_Track_detection_copy:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        filter_size = 11
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = move_speed
        self.track = []
        self.max_track_id = 0

    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                post_track = self.track[-1]
                repli_coord = self.detection_replication(post_track[:, 1:], coord)
                
                # post_track = torch.cat([post_track for _ in range(3)], dim=0)
                dist = (coord[:, None] - post_track[None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                match_flag1, dist_idx1 = linear_assignment(dist, self.move_speed[0])

                # マッチ処理
                matched_id = post_track[dist_idx1[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)

                # 現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    # マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    # マッチしていない過去の検出
                    uni_ids1 = torch.unique(self.track[-1][:, 0])
                    uni_ids2 = torch.unique(self.track[-2][:, 0])
                    unmatch_track_ids = uni_ids2[~torch.any(uni_ids2[:, None] == uni_ids1[None], dim=-1)]
                    unmatch_track = self.track[-2][torch.any(self.track[-2][:, 0, None] == unmatch_track_ids[None], dim=-1)]
                    # unmatch_track = torch.cat([unmatch_track for _ in range(3)], dim=0)

                    dist = (no_match_coord[:, None] - unmatch_track[:, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    match_flag2, dist_idx2 = linear_assignment(dist, self.move_speed[1])

                    # マッチ処理
                    matched_id = unmatch_track[dist_idx2[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    # 線形補間
                    matched_xy = unmatch_track[dist_idx2[match_flag2], 1:]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    # マッチング処理
                    self.track[-1] = torch.cat([self.track[-1], frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2

                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])
                
                
                #2frame間マッチングでもうまく行かない場合
                #合体対策:最小距離マッチング
                matched_idx = dist_idx1[~all_match_flag].to("cpu").tolist()
                unmatched_track_idx = [idx for idx in range(len(post_track)) if not idx in matched_idx]
                unmatched_track = post_track[unmatched_track_idx]
                dist = (unmatched_track[:, None] - coord[None, :, 1:]) ** 2
                minimum_dist, minimum_idx = dist.sum(dim=-1).min(dim=-1)
                minimum_match_flag = minimum_dist < self.move_speed[0] ** 2

                # マッチ処理
                matched_id = post_track[minimum_match_flag, 0]
                matched_coord = coord[minimum_idx[minimum_match_flag]]
                minimum_matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                all_match_flag[minimum_idx[minimum_match_flag]] = True


                # マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(
                    torch.cat([matched_track, matched2_track, minimum_matched_track, new_track], dim=0))

            else:  # trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = track_id[-1]


    def MPM2vector(self, outputs):
        """Function to detect the position of an object from a probability map

        Args:
            outputs (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """
        position = outputs[:, 0].flatten(0, 1)

        position = (position >= self.noise_strength) * (position == self.maxpool(position)) * 1.0
        for _ in range(5):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position == self.maxpool(position)) * 1.0

        # [detection number, 3(batch*length,y,x)]
        coordinate = torch.nonzero(position)
        # [batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(position.size(0))]

        return coordinate  # [batch*length] [num,2(x, y)]

    def detection_replication(self, post_d, present_d):
        """現在の検出物を複製する

        Args:
            post_d (torch.tensor([n,2(x,y)])): 過去の検出
            present_d (torch.tensor([m,2(x,y)])): 現在の検出

        Returns:
            torch.tensor([M,2(x,y)]): 複製された現在の検出
        """        
        distance = (post_d[:, None] - present_d[None]) ** 2
        distance = torch.sqrt(distance.sum(dim=-1))

        #1つの過去の物体から現在の物体の距離が最小のものを選ぶ
        values, idx = distance.min(dim=-1)
        match_flag = values < self.move_speed[0]
        values = values[match_flag]
        idx = idx[match_flag]

        # 最小距離の物が被ったらそのidxと被った個数を計算
        uni_idx, uni_counts = torch.unique(idx, return_counts=True)
        replication_flag = uni_counts > 1
        uni_idx = uni_idx[replication_flag]
        uni_counts = uni_counts[replication_flag]
        
        #複製
        uni_counts -= 1
        repli_d = [present_d[i, None] for i, count in zip(uni_idx, uni_counts) for _ in range(count)]
        repli_d.insert(0, present_d)
        repli_d = torch.cat(repli_d, dim=0)
        
        return repli_d

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, length, delay=1):
        track = []
        for idx in range(len(self.track)):
            if idx % delay == 0:
                frame = idx//delay * torch.ones(self.track[idx].shape[0])
                track.append(
                    torch.cat([frame[:, None], self.track[idx]], dim=-1))
        print(len(track))
        track = torch.cat(track[:length], dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > 2]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])

        
        uni_ids = torch.unique(track[:, 1])
        for idx, uid in enumerate(uni_ids):
            track[track[:, 1] == uid, 1] = idx
        print(track)
        print(track.shape)
        return track


"""
    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                post_track = self.track[-1]
                post_track = torch.cat([post_track for _ in range(3)], dim=0)
                dist = (coord[:, None] - post_track[None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                match_flag1, dist_idx = linear_assignment(dist, self.move_speed[0])

                # マッチ処理
                matched_id = post_track[dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)

                # 現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    # マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    # マッチしていない過去の検出
                    uni_ids1 = torch.unique(self.track[-1][:, 0])
                    uni_ids2 = torch.unique(self.track[-2][:, 0])
                    unmatch_track_ids = uni_ids2[~torch.any(uni_ids2[:, None] == uni_ids1[None], dim=-1)]
                    unmatch_track = self.track[-2][torch.any(self.track[-2][:, 0, None] == unmatch_track_ids[None], dim=-1)]
                    unmatch_track = torch.cat([unmatch_track for _ in range(3)], dim=0)

                    dist = (no_match_coord[:, None] - unmatch_track[:, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    match_flag2, dist_idx = linear_assignment(dist, self.move_speed[1])

                    # マッチ処理
                    matched_id = unmatch_track[dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    # 線形補間
                    matched_xy = unmatch_track[dist_idx[match_flag2], 1:]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    # マッチング処理
                    self.track[-1] = torch.cat([self.track[-1], frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])

                # マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(
                    torch.cat([matched_track, matched2_track, new_track], dim=0))

            else:  # trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = track_id[-1]

"""

# 最小値マッチング
class MPM_to_Track2:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur(
            (13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = move_speed
        self.track = []
        self.max_track_id = 0

    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                dist = (coord[:, None] - self.track[-1][None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                dist_value, dist_idx = dist.min(dim=-1)

                match_flag1 = dist_value < self.move_speed[0]  # *2.0

                # マッチ処理
                matched_id = self.track[-1][dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat(
                    [matched_id[:, None], matched_coord], dim=-1)

                # 現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    # マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    # マッチしていない過去の検出
                    self.track[-2][None, :, 1:]
                    dist = (no_match_coord[:, None] -
                            self.track[-2][None, :, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    dist_value, dist_idx = dist.min(dim=-1)

                    match_flag2 = dist_value < self.move_speed[1]*2.0

                    # マッチ処理
                    matched_id = self.track[-2][dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat(
                        [matched_id[:, None], matched_coord], dim=-1)
                    # 線形補間
                    matched_xy = self.track[-2][dist_idx[match_flag2], 1:]
                    frame1diff_track = torch.cat(
                        [matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    # マッチング処理
                    self.track[-2] = torch.cat([self.track[-2],
                                               frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])

                # マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + \
                        torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(
                    torch.cat([matched_track, matched2_track, new_track], dim=0))

            else:  # trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = track_id[-1]

    def MPM2vector(self, outputs):
        """Function to detect the position of an object from a probability map

        Args:
            outputs (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """
        position = outputs[:, 0].flatten(0, 1)

        position = (position >= self.noise_strength) * \
            (position == self.maxpool(position)) * 1.0
        for _ in range(5):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position ==
                                          self.maxpool(position)) * 1.0

        # [detection number, 3(batch*length,y,x)]
        coordinate = torch.nonzero(position)
        # [batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:]
                      for batch in range(position.size(0))]

        return coordinate  # [batch*length] [num,2(x, y)]

    def get_track(self):
        return self.track

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, length, delay=1):
        track = []
        for idx in range(len(self.track)):
            if idx % delay == 0:
                frame = idx//delay * torch.ones(self.track[idx].shape[0])
                track.append(
                    torch.cat([frame[:, None], self.track[idx]], dim=-1))
        print(len(track))
        track = torch.cat(track[:length], dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > 2]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])
        
        uni_ids = torch.unique(track[:, 1])
        for idx, uid in enumerate(uni_ids):
            track[track[:, 1] == uid, 1] = idx

        print(track)
        print(track.shape)
        return track


def global_association_solver(track):
    uni_ids = torch.unique(track[:, 1])
    uni_ids = uni_ids.long()

    first_coord = torch.stack([track[track[:, 1] == uid][0] for uid in uni_ids])
    latest_coord = torch.stack([track[track[:, 1] == uid][-1] for uid in uni_ids])

    length = torch.tensor([len(track[track[:, 1] == uid]) for uid in uni_ids], device=track.device)
    length = length.float()

    track_num = len(uni_ids)
    #前半が前、後半が後に結合するものを示す
    #自身のidで
    cost_vector = np.zeros([2 * track_num], dtype=np.uint8)

    #[H,W]
    size = [512, 512]
    process_dict = {}

    for idx, uid in enumerate(uni_ids):
        fl_time_dist = first_coord[idx: idx + 1, 0] - latest_coord[:, 0]
        fl_space_dist = first_coord[idx: idx + 1, [2, 3]] - latest_coord[:, [2, 3]]
        fl_space_dist = torch.sqrt((fl_space_dist ** 2).sum(dim=-1))

        lf_time_dist = first_coord[:, 0] - latest_coord[idx: idx + 1, 0]
        lf_space_dist = latest_coord[idx: idx + 1, [2, 3]] - first_coord[:, [2, 3]]
        lf_space_dist = torch.sqrt((lf_space_dist ** 2).sum(dim=-1))
        
        fl_connect_flag = (fl_time_dist > 0) & (fl_time_dist <= 2) & (fl_space_dist < 7.)
        lf_connect_flag = (lf_time_dist > 0) & (lf_time_dist <= 2) & (lf_space_dist < 7.)
        #自身とのつながりは無くす
        fl_connect_flag[idx] = False
        lf_connect_flag[idx] = False

        edge_dist = min([first_coord[idx, 2], first_coord[idx, 3], size[1] - first_coord[idx, 2], size[0] - first_coord[idx, 3]])
        
        #register appearance
        p = torch.exp(-edge_dist / 3.).item()
        p = max(0.05, p)
        _cost = cost_vector.copy()
        _cost[track_num + idx] = 1
        process_dict[f"init_{uid}"] = [p, _cost]


        #register disappear
        p = torch.exp(-edge_dist / 3.).item()
        p = max(0.05, p)
        _cost = cost_vector.copy()
        _cost[idx] = 1
        process_dict[f"{uid}_end"] = [p, _cost]

        #register association
        #nonzero 1-D vector => [num,1]になるため次元削減
        connect_idx = torch.nonzero(lf_connect_flag)[:, 0]
        for con_idx in connect_idx:
            p = torch.exp(-lf_space_dist[con_idx] / (7.*lf_time_dist[con_idx])).item()
            _cost = cost_vector.copy()
            _cost[idx] = 1
            _cost[track_num + con_idx] = 1
            process_dict[f"{uid}_{uni_ids[con_idx.item()]}"] = [p, _cost]
        
        #register combine
        connect_idx = torch.nonzero(fl_connect_flag)[:, 0]
        if len(connect_idx) > 1:
            combinations = torch.combinations(connect_idx, r=2)
            for con_idx in combinations:
                p = torch.exp(-fl_space_dist[con_idx] / (8.*lf_time_dist[con_idx])).sum().item()
                _cost = cost_vector.copy()
                _cost[con_idx[0]] = 1
                _cost[con_idx[1]] = 1
                _cost[track_num + idx] = 1
                process_dict[f"{uni_ids[con_idx[0].item()]},{uni_ids[con_idx[1].item()]}_{uid}"] = [p, _cost]

        #register division
        connect_idx = torch.nonzero(lf_connect_flag)[:, 0]
        if len(connect_idx) > 1:
            combinations = torch.combinations(connect_idx, r=2)
            for con_idx in combinations:
                p = torch.exp(-lf_space_dist[con_idx] / (8.*lf_time_dist[con_idx])).sum().item()
                _cost = cost_vector.copy()
                _cost[track_num + con_idx[0]] = 1
                _cost[track_num + con_idx[1]] = 1
                _cost[idx] = 1
                process_dict[f"{uid}_{uni_ids[con_idx[0].item()]},{uni_ids[con_idx[1].item()]}"] = [p, _cost]

        #register FP
        p = torch.exp(-length[idx]/1.).item()
        _cost = cost_vector.copy()
        _cost[idx] = 1
        _cost[track_num + idx] = 1
        process_dict[f"{uid}isFP"] = [p, _cost]

    #Solver
    print("calculate cost matrix")
    # [process_num]
    p_vector = np.array([v[0] for v in process_dict.values()])
    # [process_num, 2*track_num]
    cost_matrix = np.stack([v[1] for v in process_dict.values()])

    problem = LpProblem(sense=LpMaximize)
    process_flags = [LpVariable(key, cat=LpBinary) for key in process_dict.keys()]

    # maximum
    problem += lpDot(p_vector, process_flags)
    for cost_vec in tqdm(cost_matrix.T):
        problem += lpDot(cost_vec, process_flags) == 1
        # problem += lpSum(list(map(lambda cx: cx[0] * cx[1], zip(cost_vec, process_flags)))) == 1
        # problem += lpSum([c * x for c, x in zip(cost_vec, process_flags)]) == 1

    print("start solve")
    problem.solve()

    print(f"all process: {sum([1 if item.value() else 0 for item in process_flags])}")
    counts_dict = {
        "init":0,
        "connect":0,
        "combine":0,
        "division":0,
        "end":0,
        "FP":0
    }
    for item in process_flags:
        if item.value():
            if "init" in item.name:
                counts_dict["init"] += 1
            if re.match(r"\d+_\d+", item.name):
                counts_dict["connect"] += 1
            if re.match(r"\d+,\d+_\d+", item.name):
                counts_dict["combine"] += 1
            if re.match(r"\d+_\d+,\d+", item.name):
                counts_dict["division"] += 1
            if "end" in item.name:
                counts_dict["end"] += 1
            if "FP" in item.name:
                counts_dict["FP"] += 1
    for k, v in counts_dict.items():
        print(f"{k}:{v}")

    print(LpStatus[problem.status])
    print("目的関数", value(problem.objective))

    use_process = [process.name for process in process_flags if process.value() and not re.match(r"\d+isFP", process.name)]
    #初期化プロセスを抽出
    init_process = [pro[5:] for pro in use_process if re.match(r"init_\d+", pro)]
    
    #コネクトプロセスを抽出
    #消失も含む
    connect_process = []
    copro_append = connect_process.append
    for pro in use_process:
        if not re.match(r"init_\d+", pro):
            pre_pro, nxt_pro = pro.split("_")
            pre_pro_nums = re.findall(r"\d+", pre_pro)
            nxt_pro_nums = re.findall(r"(\d+|end)", nxt_pro)
            for pre in pre_pro_nums:
                for nxt in nxt_pro_nums:
                    copro_append([pre, nxt])
    
    connect_process = np.array(connect_process, dtype=str)

    track_dict = {f"{uid}": track[track[:, 1] == uid] for uid in uni_ids}
    
    result_tracks = []

    #run process
    for init_tid in init_process:
        track = make_tree_track(track_dict[init_tid], init_tid, track_dict, connect_process)
        # print(len(track))
        result_tracks.extend(track)

    for idx in range(len(result_tracks)):
        track = result_tracks[idx]
        track[:, 1] = idx
        result_tracks[idx] = track
    
    return torch.cat(result_tracks, dim=0)



def make_tree_track(track, track_id, track_dict, connect_process):
    connect_track_ids = connect_process[connect_process[:, 0] == track_id, 1]
    tracks = []
    for idx, con_id in enumerate(connect_track_ids):
        if con_id == "end":
            #endと分裂は同時に起こらないのでfor文は1回だけ
            return [track]
        nxt_track = torch.cat([track, track_dict[con_id]])
        
        tracks.extend(make_tree_track(nxt_track, con_id, track_dict, connect_process))



    return tracks
