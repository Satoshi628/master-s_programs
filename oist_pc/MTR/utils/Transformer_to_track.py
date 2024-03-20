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
from tqdm import tqdm

#----- Module -----#
# None

PAD_ID = -1
NONE_TOKEN_ID = -2
NONE_TOKEN_IDX = 0
IGNORE_NODE = -1

class Transformer_to_Track:
    def __init__(self, tracking_mode="P",
                 accuracy_rate=0.90,
                 correct_range=22,
                 move_limit=[25., 30., 35.]):
        # tracking_mode is P or N
        self.tracking_mode = tracking_mode
        self.accuracy_rate = accuracy_rate
        self.correct_range = correct_range
        self.move_limit = move_limit

        self.connect_inner = []
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = [[] for _ in range(3)]
        self.coord = []
        self.time_mode = {"P": -1, "N": 1}
        self.inf = 99999

        self.MAX_Connect_Number = 100

    def update(self, vector, coordinate):
        vector = F.normalize(vector, dim=-1).flatten(0, 1)
        vector = vector.transpose(0, 1).to('cpu').detach().clone()
        for connect_vector in self.connect_vector:
            # [batch*T,num,dim]
            connect_vector.extend([vector[:, idx] for idx in range(vector.size(1))])

        coord = coordinate.flatten(0, 1).transpose(0, 1).to('cpu').detach().clone()
        self.coord.extend([coord[:, idx] for idx in range(coord.size(1))])

    def reset(self):
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = [[] for _ in range(3)]
        self.coord = []

    def get_track(self):
        return self.track.clone()

    def save_track(self, path="track"):
        track = self.track.to('cpu').detach().numpy().copy()
        np.save(path, track)

    def __call__(self):
        # coord.size() => [frame,num,2(x,y)]
        self.coord = torch.nn.utils.rnn.pad_sequence(self.coord, batch_first=True, padding_value=PAD_ID)
        # None token=-2
        self.coord[:, 0] = NONE_TOKEN_ID

        for idx in range(len(self.connect_vector)):
            # [frame,num,dim]
            connect_vector = torch.nn.utils.rnn.pad_sequence(self.connect_vector[idx], batch_first=True)

            self.connect_vector[idx] = connect_vector

        self.make_connect()
        self.make_MLP()

        # make white list [0,,5(frame,t idx, t-1 idx,t-2 idx, t-3 idx)]
        # If not whitelisted, the node can bind to all objects.
        # If it is on the white list, then the node can only bind to objects on the white list.
        self.node_white_list = self.node.new_empty([0, 5])

        self.past_detection_error_fix()
        self.nxt_detection_error_fix()

        # White list duplicate deletion
        overlap_flag_idx = torch.all(self.node_white_list[:, None] == self.node_white_list[None], dim=-1)
        if self.node_white_list.size(0) != 0:
            overlap_flag_idx = overlap_flag_idx.max(dim=-1)[1]
            overlap_flag_idx = torch.unique(overlap_flag_idx)
            self.node_white_list = self.node_white_list[overlap_flag_idx]

        self.track = self.make_track()
        self.track = self.track.flatten(0, 1)

    def make_connect(self):
        self.connect_inner = []
        for i in range(1, 4):  # i=1,2,3
            # [frame-i,past num,present num]
            connect_inner = (self.connect_vector[i - 1][:-i, :, None]
                             * self.connect_vector[i - 1][i:, None, :]).sum(dim=-1)

            PAD_flag = self.coord[:, :, 0] == PAD_ID
            # [frame-1,past num, present num]
            PAD_flag = PAD_flag[:-i, :, None] | PAD_flag[i:, None, :]
            connect_inner[PAD_flag] = -np.inf
            
            #[frame-1,past num,present num,2]
            move_vector = self.coord[:-i, :, None] - self.coord[i:, None, :]

            # Using torch.norm makes the process slower. [frame-1,num]
            move_norm = (move_vector ** 2).sum(dim=-1)
            move_norm = torch.sqrt(move_norm)
            #True is normal
            move_limit_flag = move_norm <= self.move_limit[i - 1]
            move_limit_flag[:, 0, :] = True
            move_limit_flag[:, :, 0] = True
            connect_inner[~move_limit_flag] = -np.inf

            self.connect_inner.append(connect_inner)

    def make_MLP(self):
        # [frame-1,past num,present num]
        one_delay_connect_inner = self.connect_inner[0]

        # [frame-1,past num]
        past_present_idx = one_delay_connect_inner.max(dim=-1)[1]
        # [frame-1,present num]
        present_past_idx = one_delay_connect_inner.max(dim=-2)[1]

        past_present_idx = torch.cat([past_present_idx, past_present_idx.new_zeros([1, past_present_idx.size(1)])], dim=0)
        present_past_idx = torch.cat([present_past_idx.new_zeros([1, present_past_idx.size(1)]), present_past_idx], dim=0)

        # [frame, num]
        present_idx = torch.arange(self.connect_vector[0].size(1))[None].expand(self.connect_vector[0].size(0), -1)

        # past[-1] present[0] future[1]
        #[frame, num, 3]
        self.node = torch.stack([present_idx, past_present_idx, present_past_idx], dim=-1)

        #[frame, num, 3, MAX_Connect_Number]
        self.node = self.node.unsqueeze(-1).repeat(1, 1, 1, self.MAX_Connect_Number)

        self.node[:, :, -1, 1:] = IGNORE_NODE
        self.node[:, :, 1, 1:] = IGNORE_NODE

        PAD_flag = self.coord[:, :, 0] == PAD_ID
        PAD_flag = PAD_flag[:, :, None, None].repeat(1, 1, 3, self.MAX_Connect_Number)
        PAD_flag[:, :, 0] = False
        self.node[PAD_flag] = IGNORE_NODE

        if self.tracking_mode == "P":
            previous_connect = self.node[1:, :, -1, 0]
            targets_flag = torch.any(self.node[:-1, :, 0, 0, None] == previous_connect[:, None, :], dim=-1)

            targets_flag = torch.cat([targets_flag, targets_flag.new_zeros([1, targets_flag.shape[1]], dtype=bool)], dim=0)
            targets_flag = targets_flag[:, :, None].repeat(1, 1, 3)
            targets_flag[:, :, [-1, 0]] = False

            self.node[targets_flag] = IGNORE_NODE

        elif self.tracking_mode == "N":
            next_connect = self.node[:-1, :, 1, 0]
            targets_flag = torch.any(self.node[1:, :, 0, 0, None] == next_connect[:, None, :], dim=-1)

            targets_flag = torch.cat([targets_flag.new_zeros([1, targets_flag.shape[1]], dtype=bool), targets_flag], dim=0)
            targets_flag = targets_flag[:, :, None].repeat(1, 1, 3)
            targets_flag[:, :, [0, 1]] = False

            self.node[targets_flag] = IGNORE_NODE

    def make_track(self):
        #[frame(2), track]
        track_idx = self.get_connect_node(0, mode="N").transpose(0, 1)

        # 移動ベクトルが異常なものはtrackを削除する
        #delete_flag = self.delete_move_vector_error(track_coord)
        #track_idx = track_idx[:, ~delete_flag]
        #track_coord = track_coord[:, ~delete_flag]

        for frame in tqdm(range(1, self.coord.size(0) - 1)):
            # 新しいtrackと既存のtrackを結合
            new_track_idx = self.get_connect_node(frame, mode="N").transpose(0, 1)
            overlap_idx = torch.nonzero(track_idx[-1, :, None] == new_track_idx[0, None])

            track_idx = torch.cat([track_idx[:, overlap_idx[:, 0]], new_track_idx[1:, overlap_idx[:, 1]]], dim=0)

            # ホワイトリスト適用
            delete_track_flag = self.apply_white_list(track_idx, frame)
            track_idx = track_idx[:, ~delete_track_flag]

            # multi branch削除
            delete_track_flag = self.delete_multi_branch(track_idx)
            track_idx = track_idx[:, ~delete_track_flag]

            # same track削除
            if frame % 10 == 0:
                track_idx_size = track_idx.size(1)
                delete_flag = self.delete_same_track(track_idx)
                track_idx = track_idx[:, ~delete_flag]
                while track_idx_size != track_idx.size(1):
                    track_idx_size = track_idx.size(1)
                    delete_flag = self.delete_same_track(track_idx)
                    track_idx = track_idx[:, ~delete_flag]

        # 移動ベクトル異常削除
        #delete_flag = self.delete_move_vector_error(track_idx)
        #track_idx = track_idx[:, ~delete_flag]
        count = (track_idx != NONE_TOKEN_IDX).sum(dim=0)
        track_idx = track_idx[:, count >= 2]

        track_idx = self.delete_partial_track(track_idx)

        track_idx_size = track_idx.size(1)
        delete_flag = self.delete_same_track(track_idx)
        track_idx = track_idx[:, ~delete_flag]
        while track_idx_size != track_idx.size(1):
            track_idx_size = track_idx.size(1)
            delete_flag = self.delete_same_track(track_idx)
            track_idx = track_idx[:, ~delete_flag]
        
        track_idx_size = track_idx.size(1)
        delete_flag, distance = self.delete_all_same_track(track_idx)
        track_idx = track_idx[:, ~delete_flag]
        distance = distance[:, ~delete_flag]
        distance = distance[~delete_flag, :]
        while track_idx_size != track_idx.size(1):
            track_idx_size = track_idx.size(1)
            delete_flag, distance = self.delete_all_same_track(track_idx, distance)
            track_idx = track_idx[:, ~delete_flag]
            distance = distance[:, ~delete_flag]
            distance = distance[~delete_flag, :]
        
        # [frame,num,2(x,y)]
        track = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))

        # 整形[frame,num,4(frame,id,x,y)]
        track = torch.cat([track.new_zeros([*track.shape[:2], 2]), track], dim=-1)

        # frame追加
        track[:, :, 0] = torch.arange(1, track.size(0) + 1)[:, None]
        # id追加
        track[:, :, 1] = torch.arange(track.size(1))[None, :]
        return track

    def get_connect_node(self, frame, mode="P"):
        #mode = "P"(Previous) or "N"(Next)
        time = self.time_mode[mode]

        # [num*MAX_connect,2(target idx, other idx)]
        target_other_match = torch.stack([self.node[frame, :, 0], self.node[frame, :, time]], dim=-1).flatten(0, 1)

        # [num*MAX_connect,2(target idx, other idx)]
        other_target_match = torch.stack([self.node[frame + time, :, -time], self.node[frame + time, :, 0]], dim=-1).flatten(0, 1)
        
        match = torch.cat([target_other_match, other_target_match], dim=0)

        ignore_flag = torch.any(match == IGNORE_NODE, dim=-1)
        match = match[~ignore_flag]

        overlap_flag_idx = torch.all(match[:, None] == match[None], dim=-1)
        overlap_flag_idx = overlap_flag_idx.max(dim=-1)[1]
        overlap_flag_idx = torch.unique(overlap_flag_idx)

        match = match[overlap_flag_idx]

        return match  # [match,2(target idx, other idx)]

    def delete_detection_error_branch(self):
        next_zero_match = self.node[0, self.node[0, :, 1, 0] == 0, 0, 0]

        next_zero_match_follower = self.node[1, :, -1]
        next_zero_match_follower = next_zero_match_follower[next_zero_match_follower != -1]

        next_match_flag = torch.any(next_zero_match[:, None] == next_zero_match_follower[None], dim=-1)
        next_zero_match = next_zero_match[next_match_flag]

        self.node[0, next_zero_match, 1, 0] = IGNORE_NODE

        for frame in range(1, self.node.size(0) - 1):
            past_zero_match = self.node[frame, self.node[frame, :, -1, 0] == 0, 0, 0]
            next_zero_match = self.node[frame, self.node[frame, :, 1, 0] == 0, 0, 0]

            past_zero_match_follower = self.node[frame - 1, :, 1]
            next_zero_match_follower = self.node[frame + 1, :, -1]
            past_zero_match_follower = past_zero_match_follower[past_zero_match_follower != -1]
            next_zero_match_follower = next_zero_match_follower[next_zero_match_follower != -1]

            past_zero_match_follower = torch.any(past_zero_match[:, None] == past_zero_match_follower[None], dim=-1)
            past_zero_match = past_zero_match[past_zero_match_follower]

            next_match_flag = torch.any(next_zero_match[:, None] == next_zero_match_follower[None], dim=-1)
            next_zero_match = next_zero_match[next_match_flag]

            test = self.get_connect_node(frame, mode="N")
            flag = torch.any(test[:, 0][:, None] == next_zero_match[None], dim=-1)

            self.node[frame, past_zero_match, -1, 0] = IGNORE_NODE
            self.node[frame, next_zero_match, 1, 0] = IGNORE_NODE

            test = self.get_connect_node(frame, mode="N")
            flag = torch.any(test[:, 0][:, None] == next_zero_match[None], dim=-1)

        past_zero_match = self.node[-1, self.node[-1, :, -1, 0] == 0, 0, 0]
        past_zero_match_follower = self.node[-2, :, 1]
        past_zero_match_follower = past_zero_match_follower[past_zero_match_follower != -1]
        past_zero_match_follower = torch.any(past_zero_match[:, None] == past_zero_match_follower[None], dim=-1)
        past_zero_match = past_zero_match[past_zero_match_follower]

        self.node[-1, past_zero_match, -1, 0] = IGNORE_NODE

    def past_detection_error_fix(self):
        zero_match = self.node[2, self.node[2, :, -1, 0] == 0, 0, 0]

        PAD_flag = self.coord[2, zero_match, 0] == PAD_ID
        zero_match = zero_match[~PAD_flag]

        F2_F0_idx = self.connect_inner[1][0, :, zero_match]

        F2_F0_idx = F2_F0_idx.max(dim=0)[1]

        zero_match = zero_match[F2_F0_idx != 0]
        F2_F0_idx = F2_F0_idx[F2_F0_idx != 0]

        white_list = F2_F0_idx[:, None].repeat(1, 5)
        # frame
        white_list[:, 0] = 2
        # t-3 idx
        white_list[:, 1] = IGNORE_NODE
        # t-1 idx
        white_list[:, 3] = 0
        # t idx
        white_list[:, 4] = zero_match

        self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

        F2_F0_idx = torch.unique(F2_F0_idx)

        zero_node_P_connect = self.node[1, 0, -1]

        count = (zero_node_P_connect != IGNORE_NODE).sum()
        self.node[1, 0, -1, count: count + F2_F0_idx.size(0)] = F2_F0_idx

        for frame in range(3, self.node.size(0)):
            zero_match = self.node[frame, self.node[frame, :, -1, 0] == 0, 0, 0]

            PAD_flag = self.coord[frame, zero_match, 0] == PAD_ID
            zero_match = zero_match[~PAD_flag]

            F3_F1_idx = self.connect_inner[1][frame - 2, :, zero_match]

            F3_F1_idx = F3_F1_idx.max(dim=0)[1]

            zero_flag = F3_F1_idx == 0
            list_t_idx = zero_match[~zero_flag]
            F3_F1_idx = F3_F1_idx[~zero_flag]

            white_list = F3_F1_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame
            # t-3 idx
            white_list[:, 1] = IGNORE_NODE
            # t-1 idx
            white_list[:, 3] = 0
            # t idx
            white_list[:, 4] = list_t_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F3_F1_idx = torch.unique(F3_F1_idx)

            zero_node_P_connect = self.node[frame - 1, 0, -1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame - 1, 0, -1, count: count + F3_F1_idx.size(0)] = F3_F1_idx
            
            zero_match = zero_match[zero_flag]
            # [past idx, correct idx]
            F3_F0_idx = self.connect_inner[2][frame - 3, :, zero_match]

            F3_F0_idx = F3_F0_idx.max(dim=0)[1]

            list_t_idx = zero_match[F3_F0_idx != 0]
            F3_F0_idx = F3_F0_idx[F3_F0_idx != 0]

            white_list = F3_F0_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame
            # t-2 idx
            white_list[:, 2] = 0
            # t-1 idx
            white_list[:, 3] = 0
            # t idx
            white_list[:, 4] = list_t_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F3_F0_idx = torch.unique(F3_F0_idx)

            zero_node_P_connect = self.node[frame - 2, 0, -1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame - 2, 0, -1, count: count + F3_F0_idx.size(0)] = F3_F0_idx

    def nxt_detection_error_fix(self):
        for frame in range(self.node.size(0) - 3):
            zero_match = self.node[frame, self.node[frame, :, 1, 0] == 0, 0, 0]

            PAD_flag = self.coord[frame, zero_match, 0] == PAD_ID
            zero_match = zero_match[~PAD_flag]

            F0_F2_idx = self.connect_inner[1][frame, zero_match]

            F0_F2_idx = F0_F2_idx.max(dim=-1)[1]

            zero_flag = F0_F2_idx == 0
            t_idx = zero_match[~zero_flag]
            F0_F2_idx = F0_F2_idx[~zero_flag]

            white_list = F0_F2_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame + 2
            # t-3 idx
            white_list[:, 1] = IGNORE_NODE
            # t-2 idx
            white_list[:, 2] = t_idx
            # t-1 idx
            white_list[:, 3] = 0

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F0_F2_idx = torch.unique(F0_F2_idx)

            zero_node_P_connect = self.node[frame + 1, 0, 1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame + 1, 0, 1, count: count + F0_F2_idx.size(0)] = F0_F2_idx

            zero_match = zero_match[zero_flag]

            F0_F3_idx = self.connect_inner[2][frame, zero_match]
            F0_F3_idx = F0_F3_idx.max(dim=-1)[1]

            t_idx = zero_match[F0_F3_idx != 0]
            F0_F3_idx = F0_F3_idx[F0_F3_idx != 0]

            white_list = F0_F3_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame + 3
            # t-3 idx
            white_list[:, 1] = t_idx
            # t-2 idx
            white_list[:, 2] = 0
            # t-1 idx
            white_list[:, 3] = 0

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F0_F3_idx = torch.unique(F0_F3_idx)

            zero_node_P_connect = self.node[frame + 2, 0, 1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame + 2, 0, 1, count: count + F0_F3_idx.size(0)] = F0_F3_idx
        
        zero_match = self.node[-3, self.node[-3, :, 1, 0] == 0, 0, 0]
        PAD_flag = self.coord[-3, zero_match, 0] == PAD_ID
        zero_match = zero_match[~PAD_flag]
        
        # [past idx,correct idx]
        F0_F2_idx = self.connect_inner[1][-1, zero_match]
        F0_F2_idx = F0_F2_idx.max(dim=-1)[1]

        t_idx = zero_match[F0_F2_idx != 0]
        F0_F2_idx = F0_F2_idx[F0_F2_idx != 0]

        white_list = F0_F2_idx[:, None].repeat(1, 5)
        # frame
        white_list[:, 0] = frame + 2
        # t-3 idx
        white_list[:, 1] = IGNORE_NODE
        # t-2 idx
        white_list[:, 2] = t_idx
        # t-1 idx
        white_list[:, 3] = 0

        self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

        F0_F2_idx = torch.unique(F0_F2_idx)

        zero_node_P_connect = self.node[-1, 0, 1]

        count = (zero_node_P_connect != IGNORE_NODE).sum()
        self.node[-2, 0, 1, count: count + F0_F2_idx.size(0)] = F0_F2_idx


    def apply_white_list(self, track, frame):
        white_list = self.node_white_list[self.node_white_list[:, 0] == frame + 1]

        white_list_connect_flag = (track[None, -2:] == white_list[:, -2:, None])
        white_list_connect_flag = torch.all(white_list_connect_flag, dim=1)
        white_list_connect_flag = torch.any(white_list_connect_flag, dim=0)

        if track[-4:].size(0) == 3:
            white_list_track_flag = track[None, -3:] == white_list[:, 2:, None]
            white_list_track_all_flag = track[None, - 3:-1] == white_list[:, 2:-1, None]
        else:
            white_list_track_flag = (track[None, -4:] == white_list[:, 1:, None]) | (white_list[:, 1:, None] == IGNORE_NODE)
            white_list_track_all_flag = (track[None, -4:-1] == white_list[:, 1:-1, None]) | (white_list[:, 1:-1, None] == IGNORE_NODE)

        white_list_track_flag = torch.all(white_list_track_flag, dim=1)
        white_list_track_flag = torch.any(white_list_track_flag, dim=0)

        white_list_track_all_flag = torch.all(white_list_track_all_flag, dim=1)
        white_list_track_all_flag = torch.any(white_list_track_all_flag, dim=0)

        disappearance_flag = torch.any(track[:-2] != 0, dim=0) & (track[-2] == 0) & (track[-1] != 0)

        #True is delete
        delete_track_flag1 = white_list_connect_flag & ~white_list_track_flag
        #True is delete
        delete_track_flag2 = disappearance_flag & ~white_list_connect_flag

        delete_track_flag3 = white_list_track_all_flag & ~white_list_connect_flag
        delete_track_flag = delete_track_flag1 | delete_track_flag2 | delete_track_flag3

        return delete_track_flag

    def delete_multi_branch(self, track):
        frame = track.size(0) - 1

        # [frame,num] => [num,frame]
        track = track.transpose(0, 1)

        _, index = torch.unique(track[:, -3], return_inverse=True)
        flag = (index[:, None] == index[None, :]).sum(dim=-1)
        flag = flag > 1

        overlap = track[flag, -3:]

        flag = torch.all(overlap[:, None] == overlap[None, :], dim=-1)
        flag = flag.sum(dim=-1)
        flag = flag > 1

        overlap = overlap[~flag]

        flag = torch.all(overlap[:, None, [0, 2]] == overlap[None, :, [0, 2]], dim=-1)
        flag = flag.sum(dim=-1)
        flag = flag > 1

        overlap = overlap[flag]

        if overlap.size(0) == 0:
            return track.new_zeros([track.size(0)], dtype=bool)

        flag = torch.all(overlap[:, None, [0, 2]] == overlap[None, :, [0, 2]], dim=-1)
        flag = flag.max(dim=-1)[1]
        flag = one_hot_changer(flag, overlap.shape[0], dim=-1, bool_=True)
        flag = torch.any(flag, dim=0)

        overlap_kinds = overlap[flag][:, [0, 2]]

        split_overlap = [overlap[torch.all(overlap[:, [0, 2]] == kind, dim=-1)] for kind in overlap_kinds]

        split_overlap = torch.nn.utils.rnn.pad_sequence(split_overlap, batch_first=True)

        past_vector = self.connect_vector[0][frame - 2, self.coord[frame - 2, :, 0] != -1]
        pre_vector = self.connect_vector[0][frame - 1, self.coord[frame - 1, :, 0] != -1]
        nxt_vector = self.connect_vector[0][frame, self.coord[frame, :, 0] != -1]

        past_pre_similar = (past_vector[:, None] * pre_vector[None, :]).sum(dim=-1)
        pre_nxt_similar = (pre_vector[:, None] * nxt_vector[None, :]).sum(dim=-1)

        connect_value = past_pre_similar[split_overlap[:, :, 0], split_overlap[:, :, 1]] + pre_nxt_similar[split_overlap[:, :, 1], split_overlap[:, :, 2]]

        connect_value[torch.all(split_overlap == 0, dim=-1)] = 0.

        connect_idx = connect_value.max(dim=-1)[1]

        split_overlap = torch.gather(split_overlap, dim=1, index=connect_idx[:, None, None].expand(-1, -1, 3))
        split_overlap = split_overlap.squeeze(1)

        flag = torch.all(overlap[:, None] == split_overlap[None, :], dim=-1)
        flag = torch.any(flag, dim=-1)
        delete_track = overlap[~flag]

        flag = torch.all(track[:, None, -3:] == delete_track[None, :], dim=-1)
        flag = torch.any(flag, dim=-1)
        return flag

    def delete_same_track(self, track):
        last_idx = torch.unique(track[-1, :])
        last_idx = last_idx[last_idx != 0]

        delete_flag = track.new_zeros([track.size(1)], dtype=bool)

        for idx in last_idx:
            select_flag = track[-1, :] == idx
            select_track = track[:, select_flag]
            # [frame,num,2(x,y)]
            track_coord = torch.gather(self.coord[: select_track.size(0)], dim=1, index=select_track[:, :, None].expand(-1, -1, 2))

            track_coord[track_coord < 0] = -self.inf
            distance = (track_coord[:, :, None] - track_coord[:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

            distance = distance <= self.correct_range
            distance = distance.float().mean(dim=0)
            distance = distance >= self.accuracy_rate

            match_max_value, match_max_idx = torch.max(distance.sum(dim=-1), dim=0)

            if match_max_value == 1:
                continue

            match_flag = distance[match_max_idx]
            match_track = select_track[:, match_flag]

            match_track = match_track[:, :, None].expand(-1, -1, self.connect_vector[0].shape[-1])
            # [frame,num,dim]
            match_vector = torch.gather(self.connect_vector[0], dim=1, index=match_track)

            connect_value = (match_vector[:-1] * match_vector[1:]).sum(dim=-1)
            connect_value = connect_value.mean(dim=0)
            connect_max_idx = torch.argmax(connect_value)

            match_flag[match_flag.cumsum(dim=0) == connect_max_idx + 1] = False
            delete_track = select_track[:, match_flag]

            delete_flag_temp = torch.all(track[:, :, None] == delete_track[:, None], dim=0)
            delete_flag_temp = torch.any(delete_flag_temp, dim=-1)

            delete_flag = delete_flag | delete_flag_temp

        return delete_flag

    def delete_all_same_track(self, track, distance=None):
        if distance is None:
            # [frame,num,2(x,y)]
            track_coord = torch.gather(self.coord[: track.size(0)], dim=1, index=track[:, :, None].expand(-1, -1, 2))

            track_coord[track_coord < 0] = -self.inf

            distance = (track_coord[:, :, None] - track_coord[:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

            distance = distance <= self.correct_range
            distance = distance.float().mean(dim=0)
            distance = distance >= self.accuracy_rate

        match_max_value, match_max_idx = torch.max(distance.sum(dim=-1), dim=0)

        if match_max_value == 1:
            return track.new_zeros([track.size(1)], dtype=bool), distance

        match_flag = distance[match_max_idx]
        match_track = track[:, match_flag]

        match_track = match_track[:, :, None].expand(-1, -1, self.connect_vector[0].shape[-1])
        # [frame,num,dim]
        match_vector = torch.gather(self.connect_vector[0], dim=1, index=match_track)

        connect_value = (match_vector[:-1] * match_vector[1:]).sum(dim=-1)
        connect_value = connect_value.mean(dim=0)
        connect_max_idx = torch.argmax(connect_value)

        match_flag[match_flag.cumsum(dim=0) == connect_max_idx + 1] = False
        delete_track = track[:, match_flag]

        delete_flag = torch.all(track[:, :, None] == delete_track[:, None], dim=0)
        delete_flag = torch.any(delete_flag, dim=-1)

        return delete_flag, distance

    def delete_move_vector_error(self, track_idx):
        track_coordinate = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))
        #[frame,num,2(x,y)]

        None_flag = torch.any(track_coordinate == -2,dim=-1)

        arange = torch.arange(None_flag.shape[0])[:, None].expand(-1, None_flag.shape[1])
        
        diff_zero_flag = None_flag.int().diff(n=1, dim=0)
        increase = torch.cat([diff_zero_flag.new_zeros([1, diff_zero_flag.shape[1]]), diff_zero_flag < 0])

        None_arange_idx = (~None_flag).cumsum(dim=0)-1
        diff = (arange-None_arange_idx)*increase
        diff = (2*diff-diff.cumsum(dim=0))*increase
        None_arange_idx = (~None_flag*1.0 + diff).cumsum(dim=0)-1
        None_arange_idx = None_arange_idx.long()
        None_arange_idx[~None_flag] = arange[~None_flag]

        first_zero_flag = None_arange_idx < 0
        track_idx_min = torch.min(None_arange_idx[~first_zero_flag])
        track_idx_min = torch.full_like(None_arange_idx, track_idx_min)
        None_arange_idx[first_zero_flag] = track_idx_min[first_zero_flag]

        None_arange_idx = None_arange_idx[:, :, None].expand(-1, -1, 2)

        track_coordinate = torch.gather(track_coordinate, dim=0, index=None_arange_idx)
        move_vector = torch.diff(track_coordinate, n=1, dim=0)
        move_norm = (move_vector ** 2).sum(dim=-1)
        move_norm = torch.sqrt(move_norm)

        delete_flag = torch.any(move_norm >= self.move_limit[0], dim=0)

        return delete_flag

    def delete_partial_track(self, track_idx):
        track_num = (track_idx != NONE_TOKEN_IDX).sum(dim=0)
        same_flag = track_idx[:, :, None] == track_idx[:, None]
        # ignore None token
        same_flag = same_flag & (track_idx[:, :, None] != NONE_TOKEN_IDX)
        same_flag = same_flag.sum(dim=0) / track_num
        # ignore itself track
        same_flag = same_flag - torch.eye(same_flag.shape[0], device=same_flag.device)

        same_max_value = same_flag.max(dim=-1)[0]
        track_idx = track_idx[:, same_max_value != 1.0]
        return track_idx



class Transformer_to_Track_move:
    def __init__(self):
        self.movement = []
        self.coord = []

    def update(self, move, coordinate):
        move = move.flatten(0, 1)
        move = move.transpose(0, 1).to('cpu').detach().clone()
        for connect_vector in self.connect_vector:
            # [batch*T,num,dim]
            connect_vector.extend([move[:, idx] for idx in range(move.size(1))])

        coord = coordinate.flatten(0, 1).transpose(0, 1).to('cpu').detach().clone()
        self.coord.extend([coord[:, idx] for idx in range(coord.size(1))])

    def reset(self):
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = [[] for _ in range(3)]
        self.coord = []

    def get_track(self):
        return self.track.clone()

    def save_track(self, path="track"):
        track = self.track.to('cpu').detach().numpy().copy()
        np.save(path, track)

    def __call__(self):
        # coord.size() => [frame,num,2(x,y)]
        self.coord = torch.nn.utils.rnn.pad_sequence(self.coord, batch_first=True, padding_value=PAD_ID)
        # None token=-2
        self.coord[:, 0] = NONE_TOKEN_ID

        for idx in range(len(self.connect_vector)):
            # [frame,num,dim]
            connect_vector = torch.nn.utils.rnn.pad_sequence(self.connect_vector[idx], batch_first=True)

            self.connect_vector[idx] = connect_vector

        self.make_connect()
        self.make_MLP()

        # make white list [0,,5(frame,t idx, t-1 idx,t-2 idx, t-3 idx)]
        # If not whitelisted, the node can bind to all objects.
        # If it is on the white list, then the node can only bind to objects on the white list.
        self.node_white_list = self.node.new_empty([0, 5])

        self.past_detection_error_fix()
        self.nxt_detection_error_fix()

        # White list duplicate deletion
        overlap_flag_idx = torch.all(self.node_white_list[:, None] == self.node_white_list[None], dim=-1)
        if self.node_white_list.size(0) != 0:
            overlap_flag_idx = overlap_flag_idx.max(dim=-1)[1]
            overlap_flag_idx = torch.unique(overlap_flag_idx)
            self.node_white_list = self.node_white_list[overlap_flag_idx]

        self.track = self.make_track()
        self.track = self.track.flatten(0, 1)

    def make_connect(self):
        self.connect_inner = []
        for i in range(1, 4):  # i=1,2,3
            # [frame-i,past num,present num]
            connect_inner = (self.connect_vector[i - 1][:-i, :, None]
                             * self.connect_vector[i - 1][i:, None, :]).sum(dim=-1)

            PAD_flag = self.coord[:, :, 0] == PAD_ID
            # [frame-1,past num, present num]
            PAD_flag = PAD_flag[:-i, :, None] | PAD_flag[i:, None, :]
            connect_inner[PAD_flag] = -np.inf
            
            #[frame-1,past num,present num,2]
            move_vector = self.coord[:-i, :, None] - self.coord[i:, None, :]

            # Using torch.norm makes the process slower. [frame-1,num]
            move_norm = (move_vector ** 2).sum(dim=-1)
            move_norm = torch.sqrt(move_norm)
            #True is normal
            move_limit_flag = move_norm <= self.move_limit[i - 1]
            move_limit_flag[:, 0, :] = True
            move_limit_flag[:, :, 0] = True
            connect_inner[~move_limit_flag] = -np.inf

            self.connect_inner.append(connect_inner)

    def make_MLP(self):
        # [frame-1,past num,present num]
        one_delay_connect_inner = self.connect_inner[0]

        # [frame-1,past num]
        past_present_idx = one_delay_connect_inner.max(dim=-1)[1]
        # [frame-1,present num]
        present_past_idx = one_delay_connect_inner.max(dim=-2)[1]

        past_present_idx = torch.cat([past_present_idx, past_present_idx.new_zeros([1, past_present_idx.size(1)])], dim=0)
        present_past_idx = torch.cat([present_past_idx.new_zeros([1, present_past_idx.size(1)]), present_past_idx], dim=0)

        # [frame, num]
        present_idx = torch.arange(self.connect_vector[0].size(1))[None].expand(self.connect_vector[0].size(0), -1)

        # past[-1] present[0] future[1]
        #[frame, num, 3]
        self.node = torch.stack([present_idx, past_present_idx, present_past_idx], dim=-1)

        #[frame, num, 3, MAX_Connect_Number]
        self.node = self.node.unsqueeze(-1).repeat(1, 1, 1, self.MAX_Connect_Number)

        self.node[:, :, -1, 1:] = IGNORE_NODE
        self.node[:, :, 1, 1:] = IGNORE_NODE

        PAD_flag = self.coord[:, :, 0] == PAD_ID
        PAD_flag = PAD_flag[:, :, None, None].repeat(1, 1, 3, self.MAX_Connect_Number)
        PAD_flag[:, :, 0] = False
        self.node[PAD_flag] = IGNORE_NODE

        if self.tracking_mode == "P":
            previous_connect = self.node[1:, :, -1, 0]
            targets_flag = torch.any(self.node[:-1, :, 0, 0, None] == previous_connect[:, None, :], dim=-1)

            targets_flag = torch.cat([targets_flag, targets_flag.new_zeros([1, targets_flag.shape[1]], dtype=bool)], dim=0)
            targets_flag = targets_flag[:, :, None].repeat(1, 1, 3)
            targets_flag[:, :, [-1, 0]] = False

            self.node[targets_flag] = IGNORE_NODE

        elif self.tracking_mode == "N":
            next_connect = self.node[:-1, :, 1, 0]
            targets_flag = torch.any(self.node[1:, :, 0, 0, None] == next_connect[:, None, :], dim=-1)

            targets_flag = torch.cat([targets_flag.new_zeros([1, targets_flag.shape[1]], dtype=bool), targets_flag], dim=0)
            targets_flag = targets_flag[:, :, None].repeat(1, 1, 3)
            targets_flag[:, :, [0, 1]] = False

            self.node[targets_flag] = IGNORE_NODE

    def make_track(self):
        #[frame(2), track]
        track_idx = self.get_connect_node(0, mode="N").transpose(0, 1)

        # 移動ベクトルが異常なものはtrackを削除する
        #delete_flag = self.delete_move_vector_error(track_coord)
        #track_idx = track_idx[:, ~delete_flag]
        #track_coord = track_coord[:, ~delete_flag]

        for frame in tqdm(range(1, self.coord.size(0) - 1)):
            # 新しいtrackと既存のtrackを結合
            new_track_idx = self.get_connect_node(frame, mode="N").transpose(0, 1)
            overlap_idx = torch.nonzero(track_idx[-1, :, None] == new_track_idx[0, None])

            track_idx = torch.cat([track_idx[:, overlap_idx[:, 0]], new_track_idx[1:, overlap_idx[:, 1]]], dim=0)

            # ホワイトリスト適用
            delete_track_flag = self.apply_white_list(track_idx, frame)
            track_idx = track_idx[:, ~delete_track_flag]

            # multi branch削除
            delete_track_flag = self.delete_multi_branch(track_idx)
            track_idx = track_idx[:, ~delete_track_flag]

            # same track削除
            if frame % 10 == 0:
                track_idx_size = track_idx.size(1)
                delete_flag = self.delete_same_track(track_idx)
                track_idx = track_idx[:, ~delete_flag]
                while track_idx_size != track_idx.size(1):
                    track_idx_size = track_idx.size(1)
                    delete_flag = self.delete_same_track(track_idx)
                    track_idx = track_idx[:, ~delete_flag]

        # 移動ベクトル異常削除
        #delete_flag = self.delete_move_vector_error(track_idx)
        #track_idx = track_idx[:, ~delete_flag]
        count = (track_idx != NONE_TOKEN_IDX).sum(dim=0)
        track_idx = track_idx[:, count >= 2]

        track_idx = self.delete_partial_track(track_idx)

        track_idx_size = track_idx.size(1)
        delete_flag = self.delete_same_track(track_idx)
        track_idx = track_idx[:, ~delete_flag]
        while track_idx_size != track_idx.size(1):
            track_idx_size = track_idx.size(1)
            delete_flag = self.delete_same_track(track_idx)
            track_idx = track_idx[:, ~delete_flag]
        
        track_idx_size = track_idx.size(1)
        delete_flag, distance = self.delete_all_same_track(track_idx)
        track_idx = track_idx[:, ~delete_flag]
        distance = distance[:, ~delete_flag]
        distance = distance[~delete_flag, :]
        while track_idx_size != track_idx.size(1):
            track_idx_size = track_idx.size(1)
            delete_flag, distance = self.delete_all_same_track(track_idx, distance)
            track_idx = track_idx[:, ~delete_flag]
            distance = distance[:, ~delete_flag]
            distance = distance[~delete_flag, :]
        
        # [frame,num,2(x,y)]
        track = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))

        # 整形[frame,num,4(frame,id,x,y)]
        track = torch.cat([track.new_zeros([*track.shape[:2], 2]), track], dim=-1)

        # frame追加
        track[:, :, 0] = torch.arange(1, track.size(0) + 1)[:, None]
        # id追加
        track[:, :, 1] = torch.arange(track.size(1))[None, :]
        return track

    def get_connect_node(self, frame, mode="P"):
        #mode = "P"(Previous) or "N"(Next)
        time = self.time_mode[mode]

        # [num*MAX_connect,2(target idx, other idx)]
        target_other_match = torch.stack([self.node[frame, :, 0], self.node[frame, :, time]], dim=-1).flatten(0, 1)

        # [num*MAX_connect,2(target idx, other idx)]
        other_target_match = torch.stack([self.node[frame + time, :, -time], self.node[frame + time, :, 0]], dim=-1).flatten(0, 1)
        
        match = torch.cat([target_other_match, other_target_match], dim=0)

        ignore_flag = torch.any(match == IGNORE_NODE, dim=-1)
        match = match[~ignore_flag]

        overlap_flag_idx = torch.all(match[:, None] == match[None], dim=-1)
        overlap_flag_idx = overlap_flag_idx.max(dim=-1)[1]
        overlap_flag_idx = torch.unique(overlap_flag_idx)

        match = match[overlap_flag_idx]

        return match  # [match,2(target idx, other idx)]

    def delete_detection_error_branch(self):
        next_zero_match = self.node[0, self.node[0, :, 1, 0] == 0, 0, 0]

        next_zero_match_follower = self.node[1, :, -1]
        next_zero_match_follower = next_zero_match_follower[next_zero_match_follower != -1]

        next_match_flag = torch.any(next_zero_match[:, None] == next_zero_match_follower[None], dim=-1)
        next_zero_match = next_zero_match[next_match_flag]

        self.node[0, next_zero_match, 1, 0] = IGNORE_NODE

        for frame in range(1, self.node.size(0) - 1):
            past_zero_match = self.node[frame, self.node[frame, :, -1, 0] == 0, 0, 0]
            next_zero_match = self.node[frame, self.node[frame, :, 1, 0] == 0, 0, 0]

            past_zero_match_follower = self.node[frame - 1, :, 1]
            next_zero_match_follower = self.node[frame + 1, :, -1]
            past_zero_match_follower = past_zero_match_follower[past_zero_match_follower != -1]
            next_zero_match_follower = next_zero_match_follower[next_zero_match_follower != -1]

            past_zero_match_follower = torch.any(past_zero_match[:, None] == past_zero_match_follower[None], dim=-1)
            past_zero_match = past_zero_match[past_zero_match_follower]

            next_match_flag = torch.any(next_zero_match[:, None] == next_zero_match_follower[None], dim=-1)
            next_zero_match = next_zero_match[next_match_flag]

            test = self.get_connect_node(frame, mode="N")
            flag = torch.any(test[:, 0][:, None] == next_zero_match[None], dim=-1)

            self.node[frame, past_zero_match, -1, 0] = IGNORE_NODE
            self.node[frame, next_zero_match, 1, 0] = IGNORE_NODE

            test = self.get_connect_node(frame, mode="N")
            flag = torch.any(test[:, 0][:, None] == next_zero_match[None], dim=-1)

        past_zero_match = self.node[-1, self.node[-1, :, -1, 0] == 0, 0, 0]
        past_zero_match_follower = self.node[-2, :, 1]
        past_zero_match_follower = past_zero_match_follower[past_zero_match_follower != -1]
        past_zero_match_follower = torch.any(past_zero_match[:, None] == past_zero_match_follower[None], dim=-1)
        past_zero_match = past_zero_match[past_zero_match_follower]

        self.node[-1, past_zero_match, -1, 0] = IGNORE_NODE

    def past_detection_error_fix(self):
        zero_match = self.node[2, self.node[2, :, -1, 0] == 0, 0, 0]

        PAD_flag = self.coord[2, zero_match, 0] == PAD_ID
        zero_match = zero_match[~PAD_flag]

        F2_F0_idx = self.connect_inner[1][0, :, zero_match]

        F2_F0_idx = F2_F0_idx.max(dim=0)[1]

        zero_match = zero_match[F2_F0_idx != 0]
        F2_F0_idx = F2_F0_idx[F2_F0_idx != 0]

        white_list = F2_F0_idx[:, None].repeat(1, 5)
        # frame
        white_list[:, 0] = 2
        # t-3 idx
        white_list[:, 1] = IGNORE_NODE
        # t-1 idx
        white_list[:, 3] = 0
        # t idx
        white_list[:, 4] = zero_match

        self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

        F2_F0_idx = torch.unique(F2_F0_idx)

        zero_node_P_connect = self.node[1, 0, -1]

        count = (zero_node_P_connect != IGNORE_NODE).sum()
        self.node[1, 0, -1, count: count + F2_F0_idx.size(0)] = F2_F0_idx

        for frame in range(3, self.node.size(0)):
            zero_match = self.node[frame, self.node[frame, :, -1, 0] == 0, 0, 0]

            PAD_flag = self.coord[frame, zero_match, 0] == PAD_ID
            zero_match = zero_match[~PAD_flag]

            F3_F1_idx = self.connect_inner[1][frame - 2, :, zero_match]

            F3_F1_idx = F3_F1_idx.max(dim=0)[1]

            zero_flag = F3_F1_idx == 0
            list_t_idx = zero_match[~zero_flag]
            F3_F1_idx = F3_F1_idx[~zero_flag]

            white_list = F3_F1_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame
            # t-3 idx
            white_list[:, 1] = IGNORE_NODE
            # t-1 idx
            white_list[:, 3] = 0
            # t idx
            white_list[:, 4] = list_t_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F3_F1_idx = torch.unique(F3_F1_idx)

            zero_node_P_connect = self.node[frame - 1, 0, -1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame - 1, 0, -1, count: count + F3_F1_idx.size(0)] = F3_F1_idx
            
            zero_match = zero_match[zero_flag]
            # [past idx, correct idx]
            F3_F0_idx = self.connect_inner[2][frame - 3, :, zero_match]

            F3_F0_idx = F3_F0_idx.max(dim=0)[1]

            list_t_idx = zero_match[F3_F0_idx != 0]
            F3_F0_idx = F3_F0_idx[F3_F0_idx != 0]

            white_list = F3_F0_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame
            # t-2 idx
            white_list[:, 2] = 0
            # t-1 idx
            white_list[:, 3] = 0
            # t idx
            white_list[:, 4] = list_t_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F3_F0_idx = torch.unique(F3_F0_idx)

            zero_node_P_connect = self.node[frame - 2, 0, -1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame - 2, 0, -1, count: count + F3_F0_idx.size(0)] = F3_F0_idx

    def nxt_detection_error_fix(self):
        for frame in range(self.node.size(0) - 3):
            zero_match = self.node[frame, self.node[frame, :, 1, 0] == 0, 0, 0]

            PAD_flag = self.coord[frame, zero_match, 0] == PAD_ID
            zero_match = zero_match[~PAD_flag]

            F0_F2_idx = self.connect_inner[1][frame, zero_match]

            F0_F2_idx = F0_F2_idx.max(dim=-1)[1]

            zero_flag = F0_F2_idx == 0
            t_idx = zero_match[~zero_flag]
            F0_F2_idx = F0_F2_idx[~zero_flag]

            white_list = F0_F2_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame + 2
            # t-3 idx
            white_list[:, 1] = IGNORE_NODE
            # t-2 idx
            white_list[:, 2] = t_idx
            # t-1 idx
            white_list[:, 3] = 0

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F0_F2_idx = torch.unique(F0_F2_idx)

            zero_node_P_connect = self.node[frame + 1, 0, 1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame + 1, 0, 1, count: count + F0_F2_idx.size(0)] = F0_F2_idx

            zero_match = zero_match[zero_flag]

            F0_F3_idx = self.connect_inner[2][frame, zero_match]
            F0_F3_idx = F0_F3_idx.max(dim=-1)[1]

            t_idx = zero_match[F0_F3_idx != 0]
            F0_F3_idx = F0_F3_idx[F0_F3_idx != 0]

            white_list = F0_F3_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame + 3
            # t-3 idx
            white_list[:, 1] = t_idx
            # t-2 idx
            white_list[:, 2] = 0
            # t-1 idx
            white_list[:, 3] = 0

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            F0_F3_idx = torch.unique(F0_F3_idx)

            zero_node_P_connect = self.node[frame + 2, 0, 1]

            count = (zero_node_P_connect != IGNORE_NODE).sum()
            self.node[frame + 2, 0, 1, count: count + F0_F3_idx.size(0)] = F0_F3_idx
        
        zero_match = self.node[-3, self.node[-3, :, 1, 0] == 0, 0, 0]
        PAD_flag = self.coord[-3, zero_match, 0] == PAD_ID
        zero_match = zero_match[~PAD_flag]
        
        # [past idx,correct idx]
        F0_F2_idx = self.connect_inner[1][-1, zero_match]
        F0_F2_idx = F0_F2_idx.max(dim=-1)[1]

        t_idx = zero_match[F0_F2_idx != 0]
        F0_F2_idx = F0_F2_idx[F0_F2_idx != 0]

        white_list = F0_F2_idx[:, None].repeat(1, 5)
        # frame
        white_list[:, 0] = frame + 2
        # t-3 idx
        white_list[:, 1] = IGNORE_NODE
        # t-2 idx
        white_list[:, 2] = t_idx
        # t-1 idx
        white_list[:, 3] = 0

        self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

        F0_F2_idx = torch.unique(F0_F2_idx)

        zero_node_P_connect = self.node[-1, 0, 1]

        count = (zero_node_P_connect != IGNORE_NODE).sum()
        self.node[-2, 0, 1, count: count + F0_F2_idx.size(0)] = F0_F2_idx


    def apply_white_list(self, track, frame):
        white_list = self.node_white_list[self.node_white_list[:, 0] == frame + 1]

        white_list_connect_flag = (track[None, -2:] == white_list[:, -2:, None])
        white_list_connect_flag = torch.all(white_list_connect_flag, dim=1)
        white_list_connect_flag = torch.any(white_list_connect_flag, dim=0)

        if track[-4:].size(0) == 3:
            white_list_track_flag = track[None, -3:] == white_list[:, 2:, None]
            white_list_track_all_flag = track[None, - 3:-1] == white_list[:, 2:-1, None]
        else:
            white_list_track_flag = (track[None, -4:] == white_list[:, 1:, None]) | (white_list[:, 1:, None] == IGNORE_NODE)
            white_list_track_all_flag = (track[None, -4:-1] == white_list[:, 1:-1, None]) | (white_list[:, 1:-1, None] == IGNORE_NODE)

        white_list_track_flag = torch.all(white_list_track_flag, dim=1)
        white_list_track_flag = torch.any(white_list_track_flag, dim=0)

        white_list_track_all_flag = torch.all(white_list_track_all_flag, dim=1)
        white_list_track_all_flag = torch.any(white_list_track_all_flag, dim=0)

        disappearance_flag = torch.any(track[:-2] != 0, dim=0) & (track[-2] == 0) & (track[-1] != 0)

        #True is delete
        delete_track_flag1 = white_list_connect_flag & ~white_list_track_flag
        #True is delete
        delete_track_flag2 = disappearance_flag & ~white_list_connect_flag

        delete_track_flag3 = white_list_track_all_flag & ~white_list_connect_flag
        delete_track_flag = delete_track_flag1 | delete_track_flag2 | delete_track_flag3

        return delete_track_flag

    def delete_multi_branch(self, track):
        frame = track.size(0) - 1

        # [frame,num] => [num,frame]
        track = track.transpose(0, 1)

        _, index = torch.unique(track[:, -3], return_inverse=True)
        flag = (index[:, None] == index[None, :]).sum(dim=-1)
        flag = flag > 1

        overlap = track[flag, -3:]

        flag = torch.all(overlap[:, None] == overlap[None, :], dim=-1)
        flag = flag.sum(dim=-1)
        flag = flag > 1

        overlap = overlap[~flag]

        flag = torch.all(overlap[:, None, [0, 2]] == overlap[None, :, [0, 2]], dim=-1)
        flag = flag.sum(dim=-1)
        flag = flag > 1

        overlap = overlap[flag]

        if overlap.size(0) == 0:
            return track.new_zeros([track.size(0)], dtype=bool)

        flag = torch.all(overlap[:, None, [0, 2]] == overlap[None, :, [0, 2]], dim=-1)
        flag = flag.max(dim=-1)[1]
        flag = one_hot_changer(flag, overlap.shape[0], dim=-1, bool_=True)
        flag = torch.any(flag, dim=0)

        overlap_kinds = overlap[flag][:, [0, 2]]

        split_overlap = [overlap[torch.all(overlap[:, [0, 2]] == kind, dim=-1)] for kind in overlap_kinds]

        split_overlap = torch.nn.utils.rnn.pad_sequence(split_overlap, batch_first=True)

        past_vector = self.connect_vector[0][frame - 2, self.coord[frame - 2, :, 0] != -1]
        pre_vector = self.connect_vector[0][frame - 1, self.coord[frame - 1, :, 0] != -1]
        nxt_vector = self.connect_vector[0][frame, self.coord[frame, :, 0] != -1]

        past_pre_similar = (past_vector[:, None] * pre_vector[None, :]).sum(dim=-1)
        pre_nxt_similar = (pre_vector[:, None] * nxt_vector[None, :]).sum(dim=-1)

        connect_value = past_pre_similar[split_overlap[:, :, 0], split_overlap[:, :, 1]] + pre_nxt_similar[split_overlap[:, :, 1], split_overlap[:, :, 2]]

        connect_value[torch.all(split_overlap == 0, dim=-1)] = 0.

        connect_idx = connect_value.max(dim=-1)[1]

        split_overlap = torch.gather(split_overlap, dim=1, index=connect_idx[:, None, None].expand(-1, -1, 3))
        split_overlap = split_overlap.squeeze(1)

        flag = torch.all(overlap[:, None] == split_overlap[None, :], dim=-1)
        flag = torch.any(flag, dim=-1)
        delete_track = overlap[~flag]

        flag = torch.all(track[:, None, -3:] == delete_track[None, :], dim=-1)
        flag = torch.any(flag, dim=-1)
        return flag

    def delete_same_track(self, track):
        last_idx = torch.unique(track[-1, :])
        last_idx = last_idx[last_idx != 0]

        delete_flag = track.new_zeros([track.size(1)], dtype=bool)

        for idx in last_idx:
            select_flag = track[-1, :] == idx
            select_track = track[:, select_flag]
            # [frame,num,2(x,y)]
            track_coord = torch.gather(self.coord[: select_track.size(0)], dim=1, index=select_track[:, :, None].expand(-1, -1, 2))

            track_coord[track_coord < 0] = -self.inf
            distance = (track_coord[:, :, None] - track_coord[:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

            distance = distance <= self.correct_range
            distance = distance.float().mean(dim=0)
            distance = distance >= self.accuracy_rate

            match_max_value, match_max_idx = torch.max(distance.sum(dim=-1), dim=0)

            if match_max_value == 1:
                continue

            match_flag = distance[match_max_idx]
            match_track = select_track[:, match_flag]

            match_track = match_track[:, :, None].expand(-1, -1, self.connect_vector[0].shape[-1])
            # [frame,num,dim]
            match_vector = torch.gather(self.connect_vector[0], dim=1, index=match_track)

            connect_value = (match_vector[:-1] * match_vector[1:]).sum(dim=-1)
            connect_value = connect_value.mean(dim=0)
            connect_max_idx = torch.argmax(connect_value)

            match_flag[match_flag.cumsum(dim=0) == connect_max_idx + 1] = False
            delete_track = select_track[:, match_flag]

            delete_flag_temp = torch.all(track[:, :, None] == delete_track[:, None], dim=0)
            delete_flag_temp = torch.any(delete_flag_temp, dim=-1)

            delete_flag = delete_flag | delete_flag_temp

        return delete_flag

    def delete_all_same_track(self, track, distance=None):
        if distance is None:
            # [frame,num,2(x,y)]
            track_coord = torch.gather(self.coord[: track.size(0)], dim=1, index=track[:, :, None].expand(-1, -1, 2))

            track_coord[track_coord < 0] = -self.inf

            distance = (track_coord[:, :, None] - track_coord[:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

            distance = distance <= self.correct_range
            distance = distance.float().mean(dim=0)
            distance = distance >= self.accuracy_rate

        match_max_value, match_max_idx = torch.max(distance.sum(dim=-1), dim=0)

        if match_max_value == 1:
            return track.new_zeros([track.size(1)], dtype=bool), distance

        match_flag = distance[match_max_idx]
        match_track = track[:, match_flag]

        match_track = match_track[:, :, None].expand(-1, -1, self.connect_vector[0].shape[-1])
        # [frame,num,dim]
        match_vector = torch.gather(self.connect_vector[0], dim=1, index=match_track)

        connect_value = (match_vector[:-1] * match_vector[1:]).sum(dim=-1)
        connect_value = connect_value.mean(dim=0)
        connect_max_idx = torch.argmax(connect_value)

        match_flag[match_flag.cumsum(dim=0) == connect_max_idx + 1] = False
        delete_track = track[:, match_flag]

        delete_flag = torch.all(track[:, :, None] == delete_track[:, None], dim=0)
        delete_flag = torch.any(delete_flag, dim=-1)

        return delete_flag, distance

    def delete_move_vector_error(self, track_idx):
        track_coordinate = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))
        #[frame,num,2(x,y)]

        None_flag = torch.any(track_coordinate == -2,dim=-1)

        arange = torch.arange(None_flag.shape[0])[:, None].expand(-1, None_flag.shape[1])
        
        diff_zero_flag = None_flag.int().diff(n=1, dim=0)
        increase = torch.cat([diff_zero_flag.new_zeros([1, diff_zero_flag.shape[1]]), diff_zero_flag < 0])

        None_arange_idx = (~None_flag).cumsum(dim=0)-1
        diff = (arange-None_arange_idx)*increase
        diff = (2*diff-diff.cumsum(dim=0))*increase
        None_arange_idx = (~None_flag*1.0 + diff).cumsum(dim=0)-1
        None_arange_idx = None_arange_idx.long()
        None_arange_idx[~None_flag] = arange[~None_flag]

        first_zero_flag = None_arange_idx < 0
        track_idx_min = torch.min(None_arange_idx[~first_zero_flag])
        track_idx_min = torch.full_like(None_arange_idx, track_idx_min)
        None_arange_idx[first_zero_flag] = track_idx_min[first_zero_flag]

        None_arange_idx = None_arange_idx[:, :, None].expand(-1, -1, 2)

        track_coordinate = torch.gather(track_coordinate, dim=0, index=None_arange_idx)
        move_vector = torch.diff(track_coordinate, n=1, dim=0)
        move_norm = (move_vector ** 2).sum(dim=-1)
        move_norm = torch.sqrt(move_norm)

        delete_flag = torch.any(move_norm >= self.move_limit[0], dim=0)

        return delete_flag

    def delete_partial_track(self, track_idx):
        track_num = (track_idx != NONE_TOKEN_IDX).sum(dim=0)
        same_flag = track_idx[:, :, None] == track_idx[:, None]
        # ignore None token
        same_flag = same_flag & (track_idx[:, :, None] != NONE_TOKEN_IDX)
        same_flag = same_flag.sum(dim=0) / track_num
        # ignore itself track
        same_flag = same_flag - torch.eye(same_flag.shape[0], device=same_flag.device)

        same_max_value = same_flag.max(dim=-1)[0]
        track_idx = track_idx[:, same_max_value != 1.0]
        return track_idx


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
