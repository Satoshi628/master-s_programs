#coding: utf-8
#----- Standard Library -----#
import os
import time
import math

#----- Public Package -----#
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
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
        raise ValueError("The index number of the input tensor is larger than vector_dim.")

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


class MPM_to_Track:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = move_speed
        self.track = []
        self.max_track_id = 0


    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  #空配列じゃないなら
                #現在から過去へのマッチング
                dist = (coord[:, None] - self.track[-1][None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                match_flag1, dist_idx = linear_assignment(dist, self.move_speed[0])

                #マッチ処理
                matched_id = self.track[-1][dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)

                #現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    #マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    #マッチしていない過去の検出
                    uni_ids1 = torch.unique(self.track[-1][:, 0])
                    uni_ids2 = torch.unique(self.track[-2][:, 0])
                    unmatch_track_ids = uni_ids2[~torch.any(uni_ids2[:, None] == uni_ids1[None], dim=-1)]
                    unmatch_track = self.track[-2][torch.any(self.track[-2][:, 0, None] == unmatch_track_ids[None], dim=-1)]
                    

                    dist = (no_match_coord[:, None] - unmatch_track[:, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    match_flag2, dist_idx = linear_assignment(dist, self.move_speed[1])
                    
                    #マッチ処理
                    matched_id = unmatch_track[dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    #線形補間
                    matched_xy = unmatch_track[dist_idx[match_flag2], 1:]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    #マッチング処理
                    self.track[-1] = torch.cat([self.track[-1], frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])
                

                #マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(torch.cat([matched_track, matched2_track, new_track], dim=0))

            else:  #trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = coord.shape[0] - 1

    def MPM2vector(self, outputs):
        """Function to detect the position of an object from a probability map

        Args:
            outputs (tensor[batch,1,length,H,W]): The probability map section is [0,1]

        Returns:
            tensor[batch,length,2(x,y)]: detection coordinate
        """       
        position = outputs[:, 0].flatten(0, 1)
        
        
        position = (position >= self.noise_strength) * (position == self.maxpool(position)) * 1.0
        for _ in range(3):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position == self.maxpool(position)) * 1.0
        
        coordinate = torch.nonzero(position)  #[detection number, 3(batch*length,y,x)]
        #[batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(position.size(0))]

        return coordinate #[batch*length] [num,2(x, y)]

    def get_track(self):
        return self.track

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, delay=1):
        print(len(self.track))
        track = []
        for idx in range(len(self.track)):
            if idx % delay == 0:
                frame = idx//delay * torch.ones(self.track[idx].shape[0])
                track.append(torch.cat([frame[:, None], self.track[idx]], dim=-1))
        print(len(track))
        track = torch.cat(track, dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > 2]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])

        print(track)
        print(track.shape)
        return track


class MPM_to_Track2:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = move_speed
        self.track = []
        self.max_track_id = 0


    def update(self, outputs):
        coords = self.MPM2vector(outputs)
        for coord in coords:
            if len(self.track) != 0:  #空配列じゃないなら
                #現在から過去へのマッチング
                dist = (coord[:, None] - self.track[-1][None, :, 1:]) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))
                dist_value, dist_idx = dist.min(dim=-1)

                match_flag1 = dist_value < self.move_speed[0]#*2.0
                
                #マッチ処理
                matched_id = self.track[-1][dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                matched_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)

                #現在から過去へ2frame先のマッチング
                if len(self.track) >= 2:
                    #マッチしていない現在検出
                    no_match_coord = coord[~match_flag1]
                    #マッチしていない過去の検出
                    self.track[-2][None, :, 1:]
                    dist = (no_match_coord[:, None] - self.track[-2][None, :, 1:]) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))
                    dist_value, dist_idx = dist.min(dim=-1)

                    match_flag2 = dist_value < self.move_speed[1]*2.0
                    
                    #マッチ処理
                    matched_id = self.track[-2][dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    #線形補間
                    matched_xy = self.track[-2][dist_idx[match_flag2], 1:]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord) / 2], dim=-1)
                    #マッチング処理
                    self.track[-2] = torch.cat([self.track[-2], frame1diff_track], dim=0)

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])
                    frame1diff_track = torch.empty([0, 3])
                

                #マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(torch.cat([matched_track, matched2_track, new_track], dim=0))

            else:  #trackがない場合(最初の検出時)
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
        for _ in range(3):
            position = self.gaussian_filter(position)
            position = (position != 0) * (position == self.maxpool(position)) * 1.0
        
        coordinate = torch.nonzero(position)  #[detection number, 3(batch*length,y,x)]
        #[batch*length,num,3(batch*length,y,x)] => [batch*length,num,3(batch*length,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(position.size(0))]

        return coordinate #[batch*length] [num,2(x, y)]

    def get_track(self):
        return self.track

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, delay=1):
        track = []
        for idx in range(len(self.track)):
            if idx % delay == 0:
                frame = idx//delay * torch.ones(self.track[idx].shape[0])
                track.append(torch.cat([frame[:, None], self.track[idx]], dim=-1))
        print(len(track))
        track = torch.cat(track, dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > 1]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])

        print(track)
        print(track.shape)
        return track
