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
from scipy.interpolate import CubicSpline
#----- Module -----#
from .kalman_filter import KalmanFilter
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

# 一対一マッチング
class MPM_to_Track_normal:
    def __init__(self, noise_strength=0.4, move_speed=[4.757, 5.735]):
        self.gaussian_filter = torchvision.transforms.GaussianBlur(
            (13, 13), sigma=3)
        self.maxpool = nn.MaxPool2d(11, stride=1, padding=(11 - 1) // 2)
        self.noise_strength = noise_strength
        self.move_speed = [0.1, 0.1]
        self.track = []
        self.max_track_id = 0
        self.mean_move = torch.tensor([0, 11.40]) #x, y
        self.std_move = 1 / torch.tensor([7.9443, 14.8579]) / 5  #x, y
        # self.std_move = 1/torch.tensor([2.8186, 3.8546])  #x, y

    def update(self, coords):
        for coord in coords:
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                asso_score = self.association(coord, self.track[-1][:, 1:])
                match_flag1, dist_idx = linear_assignment(asso_score, self.move_speed[0])

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

                    asso_score = self.association(no_match_coord, unmatch_track[:, 1:])
                    match_flag2, dist_idx = linear_assignment(asso_score, self.move_speed[1])

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

    def association(self, coord1, coord2):
        #coord1 => [num, 2(x,y)]
        #coord2 => [num, 2(x,y)]
        coord1 += self.mean_move[None].to(coord1.device)
        association_score = - 0.5 * (self.std_move[None, None].to(coord1.device) * (coord1[:, None] - coord2[None]) ** 2).sum(dim=-1)
        association_score = torch.exp(association_score)
        return association_score

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


#検出なしフレームにも対応したやつ
class MPM_to_Track_kalman():
    def __init__(self, move_speed=[4.757, 5.735], init_move=[0, 11.77]):
        self.move_speed = move_speed
        self.init_move = init_move
        #[track id, x, y, score]のTensorを保存
        self.track = []
        self.max_track_id = 0
        self.kalman_dict = {}

    def update(self, coords):
        """tracking update

        Args:
            coords (list[Torch.tensor[n,2(x,y)]]): detection coordinate. n can be 0.
        """        
        for coord in coords:
            coord = coord.detach().cpu()
            #もし空配列なら
            if coord.shape[0] == 0:
                self.track.append(torch.empty([0, 3]))
                continue
            
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                track_ids = self.track[-1][:, 0]
                predict_coord = []
                for tid in track_ids:
                    mean, _ = self.kalman_dict[f"{int(tid)}"].predict()
                    predict_coord.append(mean[:2])
                
                if len(predict_coord) == 0:
                    predict_coord = torch.empty([0, 2])
                else:
                    predict_coord = torch.from_numpy(np.stack(predict_coord)[None])
                
                dist = (coord[:, None, :2] - predict_coord) ** 2
                dist = torch.sqrt(dist.sum(dim=-1))

                match_flag1, dist_idx = linear_assignment(dist, self.move_speed[0])

                # マッチ処理
                matched_id = self.track[-1][dist_idx[match_flag1], 0]
                matched_coord = coord[match_flag1]
                #カルマンフィルタの更新
                for mid, mxy in zip(matched_id, matched_coord[:, :2]):
                    self.kalman_dict[f"{int(mid)}"].update(mxy.cpu().numpy())
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
                    unmatch_track_ids = unmatch_track[:, 0]
                    predict_coord1 = []
                    predict_coord2 = []
                    for tid in unmatch_track_ids:
                        #2frame分の予測
                        mean1, _ = self.kalman_dict[f"{int(tid)}"].predict()
                        mean2, _ = self.kalman_dict[f"{int(tid)}"].predict()
                        predict_coord1.append(mean1[:2])
                        predict_coord2.append(mean2[:2])
                    if len(predict_coord1) == 0:
                        predict_coord1 = torch.empty([0, 2])
                    else:
                        predict_coord1 = torch.from_numpy(np.stack(predict_coord1))
                    if len(predict_coord2) == 0:
                        predict_coord2 = torch.empty([0, 2])
                    else:
                        predict_coord2 = torch.from_numpy(np.stack(predict_coord2)[None])

                    dist = (no_match_coord[:, None, :2] - predict_coord2) ** 2
                    dist = torch.sqrt(dist.sum(dim=-1))

                    match_flag2, dist_idx = linear_assignment(dist, self.move_speed[1])

                    # マッチ処理
                    matched_id = unmatch_track[dist_idx[match_flag2], 0]
                    matched_coord = no_match_coord[match_flag2]
                    matched2_track = torch.cat([matched_id[:, None], matched_coord], dim=-1)
                    # カルマンフィルタの結果使用
                    KF_coord = predict_coord1[dist_idx[match_flag2]]
                    matched_xy = unmatch_track[dist_idx[match_flag2], 1:3]
                    frame1diff_track = torch.cat([matched_id[:, None], (matched_xy + matched_coord[:, :2]) / 2], dim=-1)
                    # frame1diff_track = torch.cat([matched_id[:, None], KF_coord, matched_coord[:, -1:]], dim=-1)
                    # マッチング処理
                    self.track[-1] = torch.cat([self.track[-1], frame1diff_track], dim=0)
                    
                    for mid, mxy in zip(matched_id, matched_coord[:, :2]):
                        self.kalman_dict[f"{int(mid)}"].update(mxy.cpu().numpy())

                    all_match_flag = match_flag1
                    all_match_flag[all_match_flag == False] = match_flag2
                else:
                    all_match_flag = match_flag1
                    matched2_track = torch.empty([0, 3])

                # マッチングできなかった検出は新しいtrackになる
                new_coord = coord[~all_match_flag]
                if new_coord.shape[0] != 0:
                    new_id = self.max_track_id + torch.arange(new_coord.shape[0]) + 1
                    for idx, tid in enumerate(new_id):
                        self.kalman_dict[f"{int(tid)}"] = KalmanFilter(new_coord[idx, :2].cpu().numpy(), self.init_move)
                    new_track = torch.cat([new_id[:, None], new_coord], dim=-1)
                    self.max_track_id = new_id[-1]
                else:
                    new_track = torch.empty([0, 3])

                self.track.append(torch.cat([matched_track, matched2_track, new_track], dim=0))
            else:  # trackがない場合(最初の検出時)
                track_id = torch.arange(coord.shape[0])
                for idx, tid in enumerate(track_id):
                    self.kalman_dict[f"{int(tid)}"] = KalmanFilter(coord[idx, :2].cpu().numpy(), self.init_move)
                track = torch.cat([track_id[:, None], coord], dim=-1)
                self.track.append(track)
                self.max_track_id = track_id[-1]

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
    limit_speed = 10.
    uni_ids = torch.unique(track[:, 1])
    uni_ids = uni_ids.long()

    first_coord = torch.stack([track[track[:, 1] == uid][0] for uid in uni_ids])
    latest_coord = torch.stack([track[track[:, 1] == uid][-1] for uid in uni_ids])

    length = torch.tensor([len(track[track[:, 1] == uid]) for uid in uni_ids], device=track.device)
    length = length.float()
    track_move = (latest_coord[:, 2:4] - first_coord[:, 2:4]) ** 2
    track_move = torch.sqrt(track_move.sum(dim=-1)) / length

    track_num = len(uni_ids)
    #前半が前、後半が後に結合するものを示す
    #自身のidで
    cost_vector = np.zeros([2 * track_num], dtype=np.uint8)

    #[H,W]
    process_dict = {}

    for idx, uid in enumerate(uni_ids):
        fl_time_dist = first_coord[idx: idx + 1, 0] - latest_coord[:, 0]
        fl_space_dist = first_coord[idx: idx + 1, [2, 3]] - latest_coord[:, [2, 3]]
        fl_space_dist = torch.sqrt((fl_space_dist ** 2).sum(dim=-1))

        lf_time_dist = first_coord[:, 0] - latest_coord[idx: idx + 1, 0]
        lf_space_dist = latest_coord[idx: idx + 1, [2, 3]] - first_coord[:, [2, 3]]
        lf_space_dist = torch.sqrt((lf_space_dist ** 2).sum(dim=-1))
        
        # fl_connect_flag = (fl_time_dist > 0) & (fl_time_dist <= 2) & (fl_space_dist < limit_speed)
        # lf_connect_flag = (lf_time_dist > 0) & (lf_time_dist <= 2) & (lf_space_dist < limit_speed)
        
        fl_connect_flag = (fl_time_dist > 0) & (fl_space_dist < limit_speed)
        lf_connect_flag = (lf_time_dist > 0) & (lf_space_dist < limit_speed)
        #自身とのつながりは無くす
        fl_connect_flag[idx] = False
        lf_connect_flag[idx] = False

        #register appearance
        p = 0.005
        _cost = cost_vector.copy()
        _cost[track_num + idx] = 1
        process_dict[f"init_{uid}"] = [p, _cost]


        #register disappear
        p = 0.005
        _cost = cost_vector.copy()
        _cost[idx] = 1
        process_dict[f"{uid}_end"] = [p, _cost]

        #register association
        #nonzero 1-D vector => [num,1]になるため次元削減
        connect_idx = torch.nonzero(lf_connect_flag)[:, 0]
        for con_idx in connect_idx:
            p = torch.exp(-lf_space_dist[con_idx] / (limit_speed*lf_time_dist[con_idx])).item()
            _cost = cost_vector.copy()
            _cost[idx] = 1
            _cost[track_num + con_idx] = 1
            process_dict[f"{uid}_{uni_ids[con_idx.item()]}"] = [p, _cost]
        
        #register FP
        p = torch.exp(-track_move[idx] / 3.).item()
        # p = 0.
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


def TrackInterpolate(tracks):
    #track => [:, (t, id, x, y, score)] ndarray
    
    track_ids = np.unique(tracks[:, 1])

    new_tracks = []

    for tid in track_ids:
        track = tracks[tracks[:, 1] == tid]
        t = track[:, 0]
        xy = track[:, 2:4]
        CS = CubicSpline(t, xy, axis=0)
        
        start_t = track[:, 0].min()
        end_t = track[:, 0].max()

        times = np.arange(start_t, end_t, 1.0)
        line = CS(times)
        new_tracks.append(np.concatenate([times[:, None], np.full([line.shape[0], 1], tid, dtype=np.int32), line], axis=-1))

    return np.concatenate(new_tracks, axis=0)



def cubic_spline(track1, track2):
    #track => [:, (t, x, y)] ndarray
    track = np.concatenate([track1, track2], axis=0)
    t = track[:, 0]
    xy = track[:, 1:]
    CS = CubicSpline(t, xy, axis=0)

    #2つの線を補間した部分
    start_t = track1[-1, 0]
    end_t = track2[0, 0]

    delta = 0.001
    times = np.arange(start_t, end_t, delta)
    line = CS(times)
    dxy_dt = np.gradient(line, times, axis=0)
    
    dxy_dt_2 = np.gradient(dxy_dt, times, axis=0)

    R = ((dxy_dt ** 2).sum(axis=-1)) ** 1.5 / np.clip(np.abs(dxy_dt[:, 0] * dxy_dt_2[:, 1] - dxy_dt[:, 1] * dxy_dt_2[:, 0]), 1e-7, None)
    
    return np.min(R)


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
