#coding: utf-8
#----- Standard Library -----#
import os
import re
import math

#----- Public Package -----#
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from pulp import *
import lap
from scipy.interpolate import CubicSpline

#----- Module -----#
from .kalman_filter import KalmanFilter
# None


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.numel() == 0:
        return torch.zeros(cost_matrix.shape[0], device=cost_matrix.device, dtype=bool), -1 * torch.ones(cost_matrix.shape[0], device=cost_matrix.device, dtype=torch.long)

    cost = cost_matrix.detach().to("cpu").numpy()
    cost, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=thresh)
    x = torch.from_numpy(x)
    x = x.long().to(cost_matrix.device)
    match_flag = x >= 0

    return match_flag, x


#検出なしフレームにも対応したやつ
#単純な重心との距離最小化マッチング
class Tracker():
    def __init__(self, move_speed=[5., 7.]):
        self.move_speed = move_speed
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
                self.track.append(torch.empty([0, 4]))
                continue
            
            if len(self.track) != 0:  # 空配列じゃないなら
                # 現在から過去へのマッチング
                #重心同士の距離最小化マッチング
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

    def get_track(self):
        return self.track

    def reset(self):
        self.track = []
        self.max_track_id = 0

    def __call__(self, length, length_th=2):
        track = []
        for idx in range(len(self.track)):
            frame = idx * torch.ones(self.track[idx].shape[0])
            track.append(
                torch.cat([frame[:, None], self.track[idx]], dim=-1))
        
        track = torch.cat(track[:length], dim=0)
        # delete 1frame track
        uni_track_id, count = torch.unique(track[:, 1], return_counts=True)
        uni_track_id = uni_track_id[count > length_th]
        track = torch.cat([track[track[:, 1] == uni_id] for uni_id in uni_track_id])

        uni_ids = torch.unique(track[:, 1])
        for idx, uid in enumerate(uni_ids):
            track[track[:, 1] == uid, 1] = idx
        
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
        p = 0.05
        _cost = cost_vector.copy()
        _cost[track_num + idx] = 1
        process_dict[f"init_{uid}"] = [p, _cost]


        #register disappear
        p = torch.exp(-edge_dist / 3.).item()
        p = max(0.05, p)
        p = 0.05
        _cost = cost_vector.copy()
        _cost[idx] = 1
        process_dict[f"{uid}_end"] = [p, _cost]

        #register association
        #nonzero 1-D vector => [num,1]になるため次元削減
        connect_idx = torch.nonzero(lf_connect_flag)[:, 0]
        for con_idx in connect_idx:
            p = torch.exp(-lf_space_dist[con_idx] / (5.*lf_time_dist[con_idx])).item()
            _cost = cost_vector.copy()
            _cost[idx] = 1
            _cost[track_num + con_idx] = 1
            process_dict[f"{uid}_{uni_ids[con_idx.item()]}"] = [p, _cost]
        
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


#追跡によって補間された部分を作成する
#スプライン補間で生成
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
