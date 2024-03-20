#coding: utf-8
#----- 標準ライブラリ -----#
import os
import time
import math
#----- 専用ライブラリ -----#
import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import h5py
#----- 自作ライブラリ -----#
#from .loss import log_optimal_transport

PAD_ID = -1
NONE_TOKEN_ID = -2
NONE_TOKEN_IDX = 0


# 1frame delay主体で2,3frame delayは検出できなかったときの対処
# 現在から過去か、現在から未来のどちらかを主体にするtracking作成方法
# 主体にした物以外で結合されなかった物はそこからから結合させる
class Transformer_to_Track:
    def __init__(self, move_limit=[25., 30., 35.],
                 delete_track_detect_frame=1,
                 match_threshold=0.0,
                 connect_method="N"):
        self.move_limit = move_limit
        self.delete_track_detect_frame = delete_track_detect_frame
        self.match_threshold = match_threshold
        # connect_method is "P" or "N" or "B"
        self.connect_method_dict = {"P": "self", "N": "other", "B": "both"}
        if connect_method == "B":
            print(
                "UserWarning: mode\"B\" is incomplete. So this mode takes a long time to complete.")
            print("You should change to mode\"N\".")
        self.connect_method = connect_method

        self.connect_idx_list = []
        self.connect_value_list = []
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = []
        self.coord = []

        self._ignore_node_number = -1

    def update(self, vector, coordinate):
        # [batch*T,num,dim]
        vector = F.normalize(vector, dim=-1)
        vector = vector.flatten(0, 1).to('cpu').detach().clone()

        # batchサイズが異なるとpaddingできないので1つずつ配列にする
        self.connect_vector.extend([vector[frame] for frame in range(vector.size(0))])

        # 座標保存
        coord = coordinate.flatten(0, 1).to('cpu').detach().clone()
        self.coord.extend([coord[frame] for frame in range(coord.size(0))])

    def reset(self):
        self.connect_idx_list = []
        self.connect_value_list = []
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = []
        self.coord = []

    def get_track(self):
        return self.track.clone()

    def save_track(self, path="track.npy"):
        track = self.track.to('cpu').detach().numpy().copy()
        np.save(path, track)

    def __call__(self):
        # coord.size() => [frame,num,2(x,y)]
        self.coord = torch.nn.utils.rnn.pad_sequence(self.coord, batch_first=True, padding_value=PAD_ID)
        # None tokenは-2の値を持つことにする
        self.coord[:, NONE_TOKEN_IDX] = NONE_TOKEN_ID

        # 検出数の違う各フレームを同じ大きさにする
        # [frame,num,dim]
        self.connect_vector = torch.nn.utils.rnn.pad_sequence(self.connect_vector, batch_first=True)

        self.make_connect()

        self.track = self.make_track()

        # trackの形を整形
        self.track = self.track.flatten(0, 1)

    def make_connect(self):
        self.connect_idx_list = []
        self.connect_value_list = []

        for delay in range(1, 4):  # i=1,2,3
            # 内積計算(ノルム1なのでCos類似度)
            similar = (self.connect_vector[:-delay, :, None] * self.connect_vector[delay:, None]).sum(dim=-1)

            # distance restriction
            distance = (self.coord[:-delay, :, None] - self.coord[delay:, None]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))
            distance = distance < self.move_limit[delay - 1]
            # accepted None Token matching
            distance[:, NONE_TOKEN_IDX] = True
            distance[:, :, NONE_TOKEN_IDX] = True

            PAD_map = (self.coord[:-delay, :, None, 0] == PAD_ID) | (self.coord[delay:, None, :, 0] == PAD_ID)

            similar = similar.masked_fill(~distance, -9e15)
            similar = similar.masked_fill(PAD_map, -9e15)

            Next_value, Next_idx = F.softmax(similar / 0.001, dim=-1).max(dim=-1)  # 未来への結合
            Previous_value, Previous_idx = F.softmax(similar / 0.001, dim=-2).max(dim=-2)  # 過去への結合

            Next_idx[Next_value < self.match_threshold] = NONE_TOKEN_IDX
            Previous_idx[Previous_value < self.match_threshold] = NONE_TOKEN_IDX

            match_flag_map = torch.zeros_like(similar)
            match_flag_map = match_flag_map.scatter_add_(-1, Next_idx.unsqueeze(-1), Next_value.unsqueeze(-1))
            match_flag_map = match_flag_map.scatter_add_(-2, Previous_idx.unsqueeze(-2), Previous_value.unsqueeze(-2))

            match_flag_map = match_flag_map * ~PAD_map
            match_flag_map[:, NONE_TOKEN_IDX, NONE_TOKEN_IDX] = 1.0

            # [num,3(frame,correct,next)]
            connect_idx = torch.nonzero(match_flag_map)

            idx_list = [connect_idx[connect_idx[:, 0] == frame][:, [1, 2]] for frame in range(match_flag_map.shape[0])]
            value_list = [value_map[idx[:, 0], idx[:, 1]] for value_map, idx in zip(match_flag_map, idx_list)]

            self.connect_idx_list.append(idx_list)
            self.connect_value_list.append(value_list)

        # vector is no longer needed, so memory out
        self.connect_vector = []

    def make_track(self):
        track_idx = self.get_max_connect_idx(1, mode="P", overlap_targets=self.connect_method_dict[self.connect_method], delay=1)
        print(track_idx.shape)

        for frame in range(2, len(self.coord)):
            new_track_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets=self.connect_method_dict[self.connect_method], delay=1)

            track_idx = self.connect_track(track_idx, new_track_idx)
            track_idx = self.detection_error_fix_delay2(frame, track_idx)
            #track_idx = self.detection_error_fix_delay3(frame, track_idx)
            print(track_idx.shape)
        
        count = (track_idx != NONE_TOKEN_IDX).sum(dim=0)
        track_idx = track_idx[:, count >= self.delete_track_detect_frame]
        
        track_idx = self.delete_overlap_track(track_idx)
        track_idx = self.delete_partial_track(track_idx)

        print(track_idx.shape)

        # [frame,num,2(x,y)]
        track = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))

        # 整形[frame,num,4(frame,id,x,y)]
        track = torch.cat([track.new_zeros([*track.shape[:2], 2]), track], dim=-1)

        # frame追加
        track[:, :, 0] = torch.arange(1, track.size(0) + 1)[:, None]
        # id追加
        track[:, :, 1] = torch.arange(track.size(1))[None, :]
        return track

    def get_connect_value(self, frame, mode="P", delay=1):
        if frame - delay < 0 and mode == "P":
            raise ValueError(f"過去に遡れません\nframe:{frame}\tdilation:{delay}")
        if frame > len(self.connect_value_list[delay - 1]) and mode == "N":
            raise ValueError(f"未来の情報がありません\nframe:{frame}\tdilation:{delay}")

        if mode == "N":
            match = self.connect_value_list[delay - 1]
            match = match[frame]

        if mode == "P":
            match = self.connect_value_list[delay - 1]
            match = match[frame - delay]
        return match

    def get_connect_idx(self, frame, mode="P", delay=1):
        if frame - delay < 0 and mode == "P":
            raise ValueError(f"過去に遡れません\nframe:{frame}\tdilation:{delay}")
        if frame > len(self.connect_idx_list[delay - 1]) and mode == "N":
            raise ValueError(f"未来の情報がありません\nframe:{frame}\tdilation:{delay}")

        if mode == "N":
            match_idx = self.connect_idx_list[delay - 1]
            match_idx = match_idx[frame]

        if mode == "P":
            match_idx = self.connect_idx_list[delay - 1]
            match_idx = match_idx[frame - delay][:, [1, 0]]

        return match_idx.transpose(0, 1)

    def get_max_connect_idx(self, frame, mode="P", overlap_targets="other", delay=1):
        """targets="other" or "self"
        """
        connect_value = self.get_connect_value(frame, mode, delay)
        connect_idx = self.get_connect_idx(frame, mode, delay)

        if overlap_targets == "self":
            targets_idx = 0
        if overlap_targets == "other":
            targets_idx = 1
        if overlap_targets == "both":
            return connect_idx

        unique_idx = torch.unique(connect_idx[targets_idx])
        unique_idx_value = torch.zeros_like(connect_value)[None].repeat(unique_idx.shape[0], 1)
        for i, idx in enumerate(unique_idx):
            unique_idx_value[i, connect_idx[targets_idx] == idx] = connect_value[connect_idx[targets_idx] == idx]

        unique_idx = unique_idx_value.max(dim=-1)[-1]
        connect_idx = connect_idx[:, unique_idx]

        return connect_idx

    def connect_track(self, old_track_idx, new_track_idx):
        # Combine old_track_idx abd new_track_idx
        # combine matching track
        matching_old_track = old_track_idx[:, old_track_idx[-1] != NONE_TOKEN_IDX]
        # [-1] mean previous idx
        matching_new_track = new_track_idx[:, new_track_idx[-1] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_old_track[-1, :, None] == matching_new_track[-1, None])
        matching_track_idx = torch.cat(
            [matching_old_track[:, overlap_idx[:, 0]], matching_new_track[:1, overlap_idx[:, 1]]], dim=0)

        # no use old track mean disappearance
        if self.connect_method == "P":
            no_match_flag = ~torch.any(matching_old_track[-1, :, None] == matching_new_track[-1, None], dim=-1)
            disappearance_track = matching_old_track[:, no_match_flag]
            disappearance_track = torch.cat([disappearance_track, matching_new_track.new_zeros([1, disappearance_track.shape[-1]])], dim=0)
        else:
            disappearance_track = old_track_idx.new_zeros([old_track_idx.shape[0] + 1, 0])
        # no use new track mean appearance
        if self.connect_method == "N":
            no_match_flag = ~torch.any(matching_old_track[-1, :, None] == matching_new_track[-1, None], dim=-2)
            appearance_track = matching_new_track[:, no_match_flag]
            appearance_track = torch.cat([matching_old_track.new_zeros([old_track_idx.shape[0] - 1, appearance_track.shape[-1]]), appearance_track], dim=0)
        else:
            appearance_track = old_track_idx.new_zeros([old_track_idx.shape[0] + 1, 0])

        # process for track matching none token
        none_old_track_idx = old_track_idx[:, old_track_idx[-1] == NONE_TOKEN_IDX]
        # [-1] mean previous idx
        none_new_track_idx = new_track_idx[:, new_track_idx[-1] == NONE_TOKEN_IDX]

        none_old_track_idx = torch.cat([none_old_track_idx, none_old_track_idx.new_zeros([1, none_old_track_idx.shape[-1]])], dim=0)
        none_new_track_idx = torch.cat([none_new_track_idx.new_zeros([old_track_idx.shape[0], none_new_track_idx.shape[-1]]), none_new_track_idx[:1]], dim=0)

        track_idx = torch.cat([matching_track_idx, disappearance_track, appearance_track, none_old_track_idx, none_new_track_idx], dim=1)
        return track_idx

    def detection_error_fix_delay2(self, frame, track_idx):
        delay2_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="other", delay=2)

        # fix track made a mistake in next matching
        # ex) track => 3, 0, 0
        fix_flag = (track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-3] != NONE_TOKEN_IDX)

        error_track = track_idx[-3:, fix_flag]

        match_flag_map = error_track[0, :, None] == delay2_matcing_idx[1, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat([error_track[:2, overlap_idx[:, 0]], delay2_matcing_idx[:1, overlap_idx[:, 1]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[-1, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[1, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-3:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape} \Maybe move_limit is so small")

        # fix track made a mistake in previous matching
        fix_flag = (track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-1] != NONE_TOKEN_IDX)

        error_track = track_idx[-3:, fix_flag]

        delay2_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="self", delay=2)
        match_flag_map = error_track[2, :, None] == delay2_matcing_idx[0, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat([delay2_matcing_idx[1:, overlap_idx[:, 1]], error_track[1:, overlap_idx[:, 0]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(frame - 2, mode="N", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[0, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[1, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-3:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape} \
                Maybe move_limit is so small")

        return track_idx

    def detection_error_fix_delay3(self, frame, track_idx):
        if frame < 3:
            return track_idx
        delay3_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="other", delay=3)

        # fix track made a mistake in next matching
        fix_flag = (track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-3] == NONE_TOKEN_IDX) & (track_idx[-4] != NONE_TOKEN_IDX)

        error_track = track_idx[-4:, fix_flag]

        match_flag_map = error_track[0, :, None] == delay3_matcing_idx[1, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat([error_track[:3, overlap_idx[:, 0]], delay3_matcing_idx[:1, overlap_idx[:, 1]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(
            frame, mode="P", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_track_idx[-1, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[-2, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        delay1_matcing_idx = self.get_max_connect_idx(frame - 1, mode="P", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:,
                                                delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[-2, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[-3, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-4:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape}\n \
                Maybe move_limit is so small")

        delay3_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="self", delay=3)

        # fix track made a mistake in previous matching
        fix_flag = (track_idx[-3] == NONE_TOKEN_IDX) & (track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-1] != NONE_TOKEN_IDX)

        error_track = track_idx[-4:, fix_flag]

        match_flag_map = error_track[-1, :, None] == delay3_matcing_idx[0, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat([delay3_matcing_idx[1:, overlap_idx[:, 1]], error_track[1:, overlap_idx[:, 0]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(frame - 2, mode="N", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[0, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[1, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        delay1_matcing_idx = self.get_max_connect_idx(frame - 1, mode="N", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[1, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[2, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-4:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape} Maybe move_limit is so small")
        return track_idx

    def delete_overlap_track(self, track_idx):
        overlap_flag = track_idx[:, :, None] == track_idx[:, None]
        overlap_flag = overlap_flag.float().mean(dim=0) == 1.0
        # no delete itself track
        # argmax is first maximum idx
        overlap_max_idx = overlap_flag.max(dim=-1)[1]
        overlap_max_idx = torch.unique(overlap_max_idx)
        track_idx = track_idx[:, overlap_max_idx]
        return track_idx

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


# Nの場合出現なし,Pの場合消失した追跡は結果からなくなる
class Transformer_to_Track2:
    def __init__(self, move_limit=[25., 30., 35.],
                 delete_track_detect_frame=2,
                 match_threshold=0.0,
                 connect_method="P"):
        self.move_limit = move_limit
        self.delete_track_detect_frame = delete_track_detect_frame
        self.match_threshold = match_threshold
        # connect_method is "P" or "N" or "B"
        self.connect_method_dict = {"P": "self", "N": "other", "B": "both"}
        if connect_method == "B":
            print(
                "UserWarning: mode\"B\" is incomplete. So this mode takes a long time to complete.")
            print("You should change to mode\"N\".")
        self.connect_method = connect_method

        self.connect_idx_list = []
        self.connect_value_list = []
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = []
        self.coord = []

        self._ignore_node_number = -1

    def update(self, vector, coordinate):
        # [batch*T,num,dim]
        vector = F.normalize(vector, dim=-1)
        vector = vector.flatten(0, 1).to('cpu').detach().clone()

        # batchサイズが異なるとpaddingできないので1つずつ配列にする
        self.connect_vector.extend([vector[frame]
                                   for frame in range(vector.size(0))])

        # 座標保存
        coord = coordinate.flatten(0, 1).to('cpu').detach().clone()
        self.coord.extend([coord[frame] for frame in range(coord.size(0))])

    def reset(self):
        self.connect_idx_list = []
        self.connect_value_list = []
        self.track = torch.empty([0, 0, 4])
        self.connect_vector = []
        self.coord = []

    def get_track(self):
        return self.track.clone()

    def save_track(self, path="track.npy"):
        track = self.track.to('cpu').detach().numpy().copy()
        np.save(path, track)

    def __call__(self):
        # coord.size() => [frame,num,2(x,y)]
        self.coord = torch.nn.utils.rnn.pad_sequence(
            self.coord, batch_first=True, padding_value=PAD_ID)
        # None tokenは-2の値を持つことにする
        self.coord[:, NONE_TOKEN_IDX] = NONE_TOKEN_ID

        # 検出数の違う各フレームを同じ大きさにする
        # [frame,num,dim]
        self.connect_vector = torch.nn.utils.rnn.pad_sequence(
            self.connect_vector, batch_first=True)

        self.make_connect()

        self.track = self.make_track()

        # trackの形を整形
        self.track = self.track.flatten(0, 1)

    def make_connect(self):
        self.connect_idx_list = []
        self.connect_value_list = []

        for delay in range(1, 4):  # i=1,2,3
            # 内積計算(ノルム1なのでCos類似度)
            similar = (self.connect_vector[:-delay, :, None]
                       * self.connect_vector[delay:, None]).sum(dim=-1)

            # distance restriction
            distance = (self.coord[:-delay, :, None] -
                        self.coord[delay:, None]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))
            distance = distance < self.move_limit[delay - 1]
            # accepted None Token matching
            distance[:, NONE_TOKEN_IDX] = True
            distance[:, :, NONE_TOKEN_IDX] = True

            PAD_map = (self.coord[:-delay, :, None, 0] ==
                       PAD_ID) | (self.coord[delay:, None, :, 0] == PAD_ID)

            similar = similar.masked_fill(~distance, -9e15)
            similar = similar.masked_fill(PAD_map, -9e15)

            Next_value, Next_idx = F.softmax(
                similar / 0.001, dim=-1).max(dim=-1)  # 未来への結合
            Previous_value, Previous_idx = F.softmax(
                similar / 0.001, dim=-2).max(dim=-2)  # 過去への結合

            Next_idx[Next_value < self.match_threshold] = NONE_TOKEN_IDX
            Previous_idx[Previous_value <
                         self.match_threshold] = NONE_TOKEN_IDX

            match_flag_map = torch.zeros_like(similar)
            match_flag_map = match_flag_map.scatter_add_(
                -1, Next_idx.unsqueeze(-1), Next_value.unsqueeze(-1))
            match_flag_map = match_flag_map.scatter_add_(
                -2, Previous_idx.unsqueeze(-2), Previous_value.unsqueeze(-2))

            match_flag_map = match_flag_map * ~PAD_map
            match_flag_map[:, NONE_TOKEN_IDX, NONE_TOKEN_IDX] = 1.0

            # [num,3(frame,correct,next)]
            connect_idx = torch.nonzero(match_flag_map)

            idx_list = [connect_idx[connect_idx[:, 0] == frame][:, [1, 2]]
                        for frame in range(match_flag_map.shape[0])]
            value_list = [value_map[idx[:, 0], idx[:, 1]]
                          for value_map, idx in zip(match_flag_map, idx_list)]

            # delete overlap

            self.connect_idx_list.append(idx_list)
            self.connect_value_list.append(value_list)

        # vector is no longer needed, so memory out
        self.connect_vector = []

    def make_track(self):
        track_idx = self.get_max_connect_idx(1, mode="P", overlap_targets=self.connect_method_dict[self.connect_method], delay=1)
        print(track_idx.shape)

        for frame in range(2, len(self.coord)):
            new_track_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets=self.connect_method_dict[self.connect_method], delay=1)

            track_idx = self.connect_track(track_idx, new_track_idx)
            track_idx = self.detection_error_fix_delay2(frame, track_idx)
            track_idx = self.detection_error_fix_delay3(frame, track_idx)
            print(track_idx.shape)

        count = (track_idx != NONE_TOKEN_IDX).sum(dim=0)
        track_idx = track_idx[:, count > self.delete_track_detect_frame]
        print(track_idx.shape)

        # [frame,num,2(x,y)]
        track = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))

        # 整形[frame,num,4(frame,id,x,y)]
        track = torch.cat([track.new_zeros([*track.shape[:2], 2]), track], dim=-1)

        # frame追加
        track[:, :, 0] = torch.arange(1, track.size(0) + 1)[:, None]
        # id追加
        track[:, :, 1] = torch.arange(track.size(1))[None, :]
        return track

    def get_connect_value(self, frame, mode="P", delay=1):
        if frame - delay < 0 and mode == "P":
            raise ValueError(f"過去に遡れません\nframe:{frame}\tdilation:{delay}")
        if frame > len(self.connect_value_list[delay - 1]) and mode == "N":
            raise ValueError(f"未来の情報がありません\nframe:{frame}\tdilation:{delay}")

        if mode == "N":
            match = self.connect_value_list[delay - 1]
            match = match[frame]

        if mode == "P":
            match = self.connect_value_list[delay - 1]
            match = match[frame - delay]
        return match

    def get_connect_idx(self, frame, mode="P", delay=1):
        if frame - delay < 0 and mode == "P":
            raise ValueError(f"過去に遡れません\nframe:{frame}\tdilation:{delay}")
        if frame > len(self.connect_idx_list[delay - 1]) and mode == "N":
            raise ValueError(f"未来の情報がありません\nframe:{frame}\tdilation:{delay}")

        if mode == "N":
            match_idx = self.connect_idx_list[delay - 1]
            match_idx = match_idx[frame]

        if mode == "P":
            match_idx = self.connect_idx_list[delay - 1]
            match_idx = match_idx[frame - delay][:, [1, 0]]

        return match_idx.transpose(0, 1)

    def get_max_connect_idx(self, frame, mode="P", overlap_targets="other", delay=1):
        """targets="other" or "self"
        """
        connect_value = self.get_connect_value(frame, mode, delay)
        connect_idx = self.get_connect_idx(frame, mode, delay)

        if overlap_targets == "self":
            targets_idx = 0
        if overlap_targets == "other":
            targets_idx = 1
        if overlap_targets == "both":
            return connect_idx

        unique_idx = torch.unique(connect_idx[targets_idx])
        unique_idx_value = torch.zeros_like(
            connect_value)[None].repeat(unique_idx.shape[0], 1)
        for i, idx in enumerate(unique_idx):
            unique_idx_value[i, connect_idx[targets_idx] ==
                             idx] = connect_value[connect_idx[targets_idx] == idx]

        unique_idx = unique_idx_value.max(dim=-1)[-1]
        connect_idx = connect_idx[:, unique_idx]

        return connect_idx

    def connect_track(self, old_track_idx, new_track_idx):
        # Combine old_track_idx abd new_track_idx
        # combine matching track
        matching_old_track = old_track_idx[:,
                                           old_track_idx[-1] != NONE_TOKEN_IDX]
        # [-1] mean previous idx
        matching_new_track = new_track_idx[:,
                                           new_track_idx[-1] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_old_track[-1, :, None] == matching_new_track[-1, None])
        matching_track_idx = torch.cat(
            [matching_old_track[:, overlap_idx[:, 0]], matching_new_track[:1, overlap_idx[:, 1]]], dim=0)

        # process for track matching none token
        none_old_track_idx = old_track_idx[:,
                                           old_track_idx[-1] == NONE_TOKEN_IDX]
        # [-1] mean previous idx
        none_new_track_idx = new_track_idx[:,
                                           new_track_idx[-1] == NONE_TOKEN_IDX]

        none_old_track_idx = torch.cat([none_old_track_idx, none_old_track_idx.new_zeros([
                                       1, none_old_track_idx.shape[-1]])], dim=0)
        none_new_track_idx = torch.cat([none_new_track_idx.new_zeros(
            [old_track_idx.shape[0], none_new_track_idx.shape[-1]]), none_new_track_idx[:1]], dim=0)

        track_idx = torch.cat(
            [matching_track_idx, none_old_track_idx, none_new_track_idx], dim=1)
        return track_idx

    def detection_error_fix_delay2(self, frame, track_idx):
        delay2_matcing_idx = self.get_max_connect_idx(
            frame, mode="P", overlap_targets="other", delay=2)

        # fix track made a mistake in next matching
        # ex) track => 3, 0, 0
        fix_flag = (
            track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-3] != NONE_TOKEN_IDX)

        error_track = track_idx[-3:, fix_flag]

        match_flag_map = error_track[0, :, None] == delay2_matcing_idx[1, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat(
            [error_track[:2, overlap_idx[:, 0]], delay2_matcing_idx[:1, overlap_idx[:, 1]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(
            frame, mode="P", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:,
                                                delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_track_idx[-1, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[1, overlap_idx[:, 0]
                           ] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-3:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape} \
                Maybe move_limit is so small")

        # fix track made a mistake in previous matching
        fix_flag = (
            track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-1] != NONE_TOKEN_IDX)

        error_track = track_idx[-3:, fix_flag]

        delay2_matcing_idx = self.get_max_connect_idx(
            frame, mode="P", overlap_targets="self", delay=2)
        match_flag_map = error_track[2, :, None] == delay2_matcing_idx[0, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat(
            [delay2_matcing_idx[1:, overlap_idx[:, 1]], error_track[1:, overlap_idx[:, 0]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(
            frame - 2, mode="N", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:,
                                                delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_track_idx[0, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[1, overlap_idx[:, 0]
                           ] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-3:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape} \
                Maybe move_limit is so small")

        return track_idx

    def detection_error_fix_delay3(self, frame, track_idx):
        if frame < 3:
            return track_idx
        delay3_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="other", delay=3)

        # fix track made a mistake in next matching
        fix_flag = (track_idx[-2] == NONE_TOKEN_IDX) & (track_idx[-3] == NONE_TOKEN_IDX) & (track_idx[-4] != NONE_TOKEN_IDX)

        error_track = track_idx[-4:, fix_flag]

        match_flag_map = error_track[0, :, None] == delay3_matcing_idx[1, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat(
            [error_track[:3, overlap_idx[:, 0]], delay3_matcing_idx[:1, overlap_idx[:, 1]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(frame, mode="P", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[-1, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[-2, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        delay1_matcing_idx = self.get_max_connect_idx(frame - 1, mode="P", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:, delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(matching_track_idx[-2, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[-3, overlap_idx[:, 0]] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-4:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape}\n \
                Maybe move_limit is so small")

        delay3_matcing_idx = self.get_max_connect_idx(
            frame, mode="P", overlap_targets="self", delay=3)

        # fix track made a mistake in previous matching
        fix_flag = (track_idx[-3] == NONE_TOKEN_IDX) & (track_idx[-2]
                                                        == NONE_TOKEN_IDX) & (track_idx[-1] != NONE_TOKEN_IDX)

        error_track = track_idx[-4:, fix_flag]

        match_flag_map = error_track[-1, :,
                                     None] == delay3_matcing_idx[0, None]
        # update fix_flag
        fix_flag[fix_flag.clone()] = torch.any(match_flag_map, dim=-1)
        overlap_idx = torch.nonzero(match_flag_map)
        matching_track_idx = torch.cat(
            [delay3_matcing_idx[1:, overlap_idx[:, 1]], error_track[1:, overlap_idx[:, 0]]], dim=0)

        # correction track
        delay1_matcing_idx = self.get_max_connect_idx(
            frame - 2, mode="N", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:,
                                                delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_track_idx[0, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[1, overlap_idx[:, 0]
                           ] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        delay1_matcing_idx = self.get_max_connect_idx(
            frame - 1, mode="N", overlap_targets="self", delay=1)
        delay1_matcing_idx = delay1_matcing_idx[:,
                                                delay1_matcing_idx[0] != NONE_TOKEN_IDX]
        overlap_idx = torch.nonzero(
            matching_track_idx[1, :, None] == delay1_matcing_idx[0, None])
        matching_track_idx[2, overlap_idx[:, 0]
                           ] = delay1_matcing_idx[1, overlap_idx[:, 1]]

        try:
            track_idx[-4:, fix_flag] = matching_track_idx
        except:
            raise ValueError(f"Different: Number of True {fix_flag.sum()} shape of matching_track_idx {matching_track_idx.shape} \
                Maybe move_limit is so small")
        return track_idx


# 1frame delay主体で2,3,4frame delayは検出できなかったときの対処
# MLP構造で処理をする
# 現在から過去か、現在から未来のどちらかを主体にするtracking作成方法
# 主体にした物以外で結合されなかった物はそこからから結合させる
class Transformer_to_Track3:
    def __init__(self, tracking_mode="P",
                 accuracy_rate=0.95,
                 correct_range=22,
                 move_limit=[25., 30., 35.]):
        """tracking_mode is P or N
        Pの場合　現在から過去への関連付けを重視し、分裂を重視するtrackになる
        Nの場合　現在から未来への関連付けを重視し、合体を重視するtrackになる"""
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
        self._None_token_coord = -2
        self._PAD_coord = -1
        self._ignore_node_number = -1

    def update(self, vector, coordinate):
        # 関連性計算
        vector = F.normalize(vector, dim=-1).flatten(0, 1)
        vector = vector.transpose(0, 1).to('cpu').detach().clone()
        for connect_vector in self.connect_vector:
            # [batch*T,num,dim]
            # batchサイズが異なるとpaddingできないので1つずつ配列にする
            connect_vector.extend([vector[:, idx]
                                  for idx in range(vector.size(1))])

        # 座標保存
        coord = coordinate.flatten(0, 1).transpose(
            0, 1).to('cpu').detach().clone()
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
        self.coord = torch.nn.utils.rnn.pad_sequence(
            self.coord, batch_first=True, padding_value=self._PAD_coord)
        # None tokenは-2の値を持つことにする
        self.coord[:, 0] = self._None_token_coord

        for idx in range(len(self.connect_vector)):
            # 検出数の違う各フレームを同じ大きさにする
            # [frame,num,dim]
            connect_vector = torch.nn.utils.rnn.pad_sequence(
                self.connect_vector[idx], batch_first=True)

            # 検出オブジェクトのvectorを保存(3つ)
            self.connect_vector[idx] = connect_vector

        self.make_connect()
        self.make_MLP()

        # 配列番号frameとframe-1との結合に関するホワイトリスト作成 [0,,5(frame,t idx, t-1 idx,t-2 idx, t-3 idx)]
        # ignore nunber は-1
        # ホワイトリストに何も登録されていない結合はすべてを受け入れる
        # ホワイトリストに登録されている結合はその結合方法のみを受け入れる
        # またホワイトリストに対象として登録された追跡はそれ以外の結合をしなくなる
        self.node_white_list = self.node.new_empty([0, 5])

        # ゼロマッチしたが、他の検出からの関連があるもののゼロマッチを削除する機構
        # 試運転できず保留
        # self.delete_detection_error_branch()

        # ゼロマッチしたものを修正する関数
        # ホワイトリストに登録され、ホワイトリストの数が修正数と一致する
        self.past_detection_error_fix()
        self.nxt_detection_error_fix()

        # ホワイトリストの重複をなくす
        overlap_flag_idx = torch.all(
            self.node_white_list[:, None] == self.node_white_list[None], dim=-1)
        if self.node_white_list.size(0) != 0:  # sizeが0だと max でエラーがでるため
            overlap_flag_idx = overlap_flag_idx.max(dim=-1)[1]
            # idxでかぶっているものを削除
            overlap_flag_idx = torch.unique(overlap_flag_idx)
            self.node_white_list = self.node_white_list[overlap_flag_idx]

        # self.Debag_get_Combination()
        # self.Debag_PAD_not_have_connect()
        self.track = self.make_track()

        # trackの形を整形
        self.track = self.track.flatten(0, 1)

    def make_connect(self):
        # connectを生成する関数
        # 内積,PAD処理,距離処理のみ行う
        # 使用するときはsoftmaxなどを使うこと
        # self.connect_inner[i][frame]としようするとframeとframe+i+1の類似度をみることができる i =0,1,2
        self.connect_inner = []
        for i in range(1, 4):  # i=1,2,3
            # [frame-i,past num,present num]
            # 1frame差の結合類似度を計算
            connect_inner = (self.connect_vector[i - 1][:-i, :, None]
                             * self.connect_vector[i - 1][i:, None, :]).sum(dim=-1)

            # PADとは関連しないようにし、またPADはNone tokenと関連するようにする
            PAD_flag = self.coord[:, :, 0] == self._PAD_coord
            # [frame-1,past num, present num]
            PAD_flag = PAD_flag[:-i, :, None] | PAD_flag[i:, None, :]
            connect_inner[PAD_flag] = -np.inf

            # 距離の限界を設定する。移動量が多すぎる物は考慮させないようにする
            # 移動ベクトルの計算[frame-1,past num,present num,2]
            move_vector = self.coord[:-i, :, None] - self.coord[i:, None, :]

            # 移動ベクトルの大きさを計算
            #torch.normを使うと処理が遅くなる [frame-1,num]
            move_norm = (move_vector ** 2).sum(dim=-1)
            move_norm = torch.sqrt(move_norm)
            #True is normal
            move_limit_flag = move_norm <= self.move_limit[i - 1]
            # None tokenへのマッチは許す
            move_limit_flag[:, 0, :] = True
            move_limit_flag[:, :, 0] = True
            connect_inner[~move_limit_flag] = -np.inf

            self.connect_inner.append(connect_inner)

    def make_MLP(self):
        # [frame-1,past num,present num]
        # 1frame差の結合類似度を取得
        one_delay_connect_inner = self.connect_inner[0]

        # 過去から現在への結合idx [frame-1,past num]
        past_present_idx = one_delay_connect_inner.max(dim=-1)[1]
        # 現在から過去への結合idx [frame-1,present num]
        present_past_idx = one_delay_connect_inner.max(dim=-2)[1]

        # 過去から現在への結合で最後尾のframeに0を付け足す
        past_present_idx = torch.cat([past_present_idx, past_present_idx.new_zeros([1, past_present_idx.size(1)])], dim=0)
        # 現在から過去への結合で最前列のframeに0を付け足す
        present_past_idx = torch.cat([present_past_idx.new_zeros([1, present_past_idx.size(1)]), present_past_idx], dim=0)

        # 過去への結合、現在のidx、未来への結合を1つにまとめる
        #現在のidxを作成 [frame, num]
        present_idx = torch.arange(self.connect_vector[0].size(1))[None].expand(self.connect_vector[0].size(0), -1)

        # [現在idx,未来idx,過去idx]の順で保存
        # 参照したいときは 過去[-1] 現在[0] 未来[1]と参照するとよい
        #[frame, num, 3]
        self.node = torch.stack(
            [present_idx, past_present_idx, present_past_idx], dim=-1)

        # 1つのノードにつき、結合はMAX_Connect_Numberまで許される
        #[frame, num, 3, MAX_Connect_Number]
        self.node = self.node.unsqueeze(-1).repeat(1, 1, 1, self.MAX_Connect_Number)

        # ignore numberは-1 現在のidx以外をignore numberで埋める
        self.node[:, :, -1, 1:] = self._ignore_node_number
        self.node[:, :, 1, 1:] = self._ignore_node_number

        # PAD自身からの関連はすべて-1にする
        PAD_flag = self.coord[:, :, 0] == self._PAD_coord
        PAD_flag = PAD_flag[:, :, None, None].repeat(
            1, 1, 3, self.MAX_Connect_Number)
        # idは削除しない
        PAD_flag[:, :, 0] = False
        self.node[PAD_flag] = self._ignore_node_number

        # track modeによって過去への関連 or 未来への関連のどちらかを削除する
        if self.tracking_mode == "P":
            # 過去への関連を残し、過去への関連の対象になった検出の未来への関連を削除する
            previous_connect = self.node[1:, :, -1, 0]
            # 過去への関連の対象になったものを抽出
            targets_flag = torch.any(self.node[:-1, :, 0, 0, None] == previous_connect[:, None, :], dim=-1)

            # 形を[frame-1,num]から[frame,num,3]に整形
            targets_flag = torch.cat([targets_flag, targets_flag.new_zeros([1, targets_flag.shape[1]], dtype=bool)], dim=0)
            targets_flag = targets_flag[:, :, None].repeat(1, 1, 3)
            # id、過去への関連は削除しない
            targets_flag[:, :, [-1, 0]] = False

            # 過去への関連の対象になったものの未来への関連を削除
            self.node[targets_flag] = self._ignore_node_number

        elif self.tracking_mode == "N":
            # 未来への関連を残し、未来への関連の対象になった検出の過去への関連を削除する
            next_connect = self.node[:-1, :, 1, 0]
            # 未来への関連の対象になったものを抽出
            targets_flag = torch.any(self.node[1:, :, 0, 0, None] == next_connect[:, None, :], dim=-1)

            # 形を[frame-1,num]から[frame,num,3]に整形
            targets_flag = torch.cat([targets_flag.new_zeros([1, targets_flag.shape[1]], dtype=bool), targets_flag], dim=0)
            targets_flag = targets_flag[:, :, None].repeat(1, 1, 3)
            # id、過去への関連は削除しない
            targets_flag[:, :, [0, 1]] = False

            # 未来への関連の対象になったものの過去への関連を削除
            self.node[targets_flag] = self._ignore_node_number

    def make_track(self):
        #[frame(2), track]
        track_idx = self.get_connect_node(0, mode="N").transpose(0, 1)

        # 移動ベクトルが異常なものはtrackを削除する
        #delete_flag = self.delete_move_vector_error(track_coord)
        #track_idx = track_idx[:, ~delete_flag]
        #track_coord = track_coord[:, ~delete_flag]

        for frame in range(1, self.coord.size(0) - 1):
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
            if frame % 20 == 0:
                track_idx_size = track_idx.size(1)
                delete_flag = self.delete_same_track(track_idx)
                track_idx = track_idx[:, ~delete_flag]
                while track_idx_size != track_idx.size(1):
                    track_idx_size = track_idx.size(1)
                    delete_flag = self.delete_same_track(track_idx)
                    track_idx = track_idx[:, ~delete_flag]
            print(track_idx.shape)

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
            print(track_idx.shape)
        
        print(track_idx.shape)
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
        # 前(Previous)か先(Next)かの変数を取得
        time = self.time_mode[mode]

        # 対象からそれ以外のマッチング [num*MAX_connect,2(target idx, other idx)]
        target_other_match = torch.stack(
            [self.node[frame, :, 0], self.node[frame, :, time]], dim=-1).flatten(0, 1)

        # それ以外から対象のマッチング [num*MAX_connect,2(target idx, other idx)]
        other_target_match = torch.stack(
            [self.node[frame + time, :, -time], self.node[frame + time, :, 0]], dim=-1).flatten(0, 1)

        # 二つを結合
        match = torch.cat([target_other_match, other_target_match], dim=0)

        # ignore判定
        ignore_flag = torch.any(match == self._ignore_node_number, dim=-1)
        match = match[~ignore_flag]

        # 重複削除
        overlap_flag_idx = torch.all(match[:, None] == match[None], dim=-1)
        # 一番初めのidxを取得
        overlap_flag_idx = overlap_flag_idx.max(dim=-1)[1]
        # idxでかぶっているものを削除
        overlap_flag_idx = torch.unique(overlap_flag_idx)

        match = match[overlap_flag_idx]

        return match  # [match,2(target idx, other idx)]

    def delete_detection_error_branch(self):
        # 現在から未来のゼロマッチング
        next_zero_match = self.node[0, self.node[0, :, 1, 0] == 0, 0, 0]

        # 未来から現在へのマッチングを検索
        next_zero_match_follower = self.node[1, :, -1]
        # ignore削除
        next_zero_match_follower = next_zero_match_follower[next_zero_match_follower != -1]

        # next_zero_match内で他の検出とマッチングしているものを抽出
        next_match_flag = torch.any(
            next_zero_match[:, None] == next_zero_match_follower[None], dim=-1)
        next_zero_match = next_zero_match[next_match_flag]

        # self.nodeのdim1はnode idxと同じ並びになっている
        self.node[0, next_zero_match, 1, 0] = self._ignore_node_number

        for frame in range(1, self.node.size(0) - 1):
            # 現在から過去のゼロマッチング
            past_zero_match = self.node[frame,
                                        self.node[frame, :, -1, 0] == 0, 0, 0]
            # 現在から未来のゼロマッチング
            next_zero_match = self.node[frame,
                                        self.node[frame, :, 1, 0] == 0, 0, 0]

            # 過去から現在、未来から現在へのマッチングを検索
            past_zero_match_follower = self.node[frame - 1, :, 1]
            next_zero_match_follower = self.node[frame + 1, :, -1]
            # ignore削除
            past_zero_match_follower = past_zero_match_follower[past_zero_match_follower != -1]
            next_zero_match_follower = next_zero_match_follower[next_zero_match_follower != -1]

            # next_zero_match内で他の検出とマッチングしているものを抽出
            past_zero_match_follower = torch.any(
                past_zero_match[:, None] == past_zero_match_follower[None], dim=-1)
            past_zero_match = past_zero_match[past_zero_match_follower]

            next_match_flag = torch.any(
                next_zero_match[:, None] == next_zero_match_follower[None], dim=-1)
            next_zero_match = next_zero_match[next_match_flag]

            test = self.get_connect_node(frame, mode="N")
            flag = torch.any(test[:, 0][:, None] == next_zero_match[None], dim=-1)
            print(test[flag])
            print(next_zero_match)

            # self.nodeのdim1はnode idxと同じ並びになっている
            self.node[frame, past_zero_match, -1, 0] = self._ignore_node_number
            self.node[frame, next_zero_match, 1, 0] = self._ignore_node_number

            test = self.get_connect_node(frame, mode="N")
            flag = torch.any(test[:, 0][:, None] ==
                             next_zero_match[None], dim=-1)
            print(test[flag])
            print(next_zero_match)
            input()

        # 現在から過去のゼロマッチング
        past_zero_match = self.node[-1, self.node[-1, :, -1, 0] == 0, 0, 0]

        # 過去から現在、未来から現在へのマッチングを検索
        past_zero_match_follower = self.node[-2, :, 1]
        # ignore削除
        past_zero_match_follower = past_zero_match_follower[past_zero_match_follower != -1]

        # next_zero_match内で他の検出とマッチングしているものを抽出
        past_zero_match_follower = torch.any(
            past_zero_match[:, None] == past_zero_match_follower[None], dim=-1)
        past_zero_match = past_zero_match[past_zero_match_follower]

        # self.nodeのdim1はnode idxと同じ並びになっている
        self.node[-1, past_zero_match, -1, 0] = self._ignore_node_number

    def past_detection_error_fix(self):
        # 対象frameからゼロにマッチングしたもの
        # 現在から過去へのマッチング
        # ignore判定も同時に行っている
        # ゼロとマッチングする結合線を-1(現在から過去)に追加することはないとする
        zero_match = self.node[2, self.node[2, :, -1, 0] == 0, 0, 0]

        # PAD削除
        PAD_flag = self.coord[2, zero_match, 0] == self._PAD_coord
        zero_match = zero_match[~PAD_flag]

        # 確信度を計算するためのベクトルを取得 [past idx,correct idx]
        F2_F0_idx = self.connect_inner[1][0, :, zero_match]

        # ゼロマッチングした検出が選ぶF0のidxを取得
        F2_F0_idx = F2_F0_idx.max(dim=0)[1]

        # ゼロ削除
        zero_match = zero_match[F2_F0_idx != 0]
        F2_F0_idx = F2_F0_idx[F2_F0_idx != 0]

        # ホワイトリスト登録
        white_list = F2_F0_idx[:, None].repeat(1, 5)
        # frame
        white_list[:, 0] = 2
        # t-3 idx
        white_list[:, 1] = self._ignore_node_number
        # t-2 idx 既にしているため実行しなくてよい
        #white_list[:, 2] = F2_F0_idx
        # t-1 idx
        white_list[:, 3] = 0
        # t idx
        white_list[:, 4] = zero_match

        self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

        # 重複削除
        F2_F0_idx = torch.unique(F2_F0_idx)

        # zeroノードへ書き込み
        # [MAX_Connect_Number]
        zero_node_P_connect = self.node[1, 0, -1]

        count = (zero_node_P_connect != self._ignore_node_number).sum()
        self.node[1, 0, -1, count: count + F2_F0_idx.size(0)] = F2_F0_idx

        for frame in range(3, self.node.size(0)):
            # 対象frameからゼロにマッチングしたもの
            # ignore判定も同時に行っている
            # ゼロとマッチングする結合線を-1(現在から過去)に追加することはないとする
            zero_match = self.node[frame, self.node[frame, :, -1, 0] == 0, 0, 0]

            # PAD削除
            PAD_flag = self.coord[frame, zero_match, 0] == self._PAD_coord
            zero_match = zero_match[~PAD_flag]

            # 確信度を計算するためのベクトルを取得 [past idx,correct idx]
            F3_F1_idx = self.connect_inner[1][frame - 2, :, zero_match]

            # ゼロマッチングした検出が選ぶF1のidxを取得
            F3_F1_idx = F3_F1_idx.max(dim=0)[1]

            # ゼロ削除,ゼロの位置は保存する
            zero_flag = F3_F1_idx == 0
            list_t_idx = zero_match[~zero_flag]
            F3_F1_idx = F3_F1_idx[~zero_flag]

            # ホワイトリスト登録
            white_list = F3_F1_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame
            # t-3 idx
            white_list[:, 1] = self._ignore_node_number
            # t-2 idx 既にしているため実行しなくてよい
            #white_list[:, 2] = F3_F1_idx
            # t-1 idx
            white_list[:, 3] = 0
            # t idx
            white_list[:, 4] = list_t_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            # 重複削除
            F3_F1_idx = torch.unique(F3_F1_idx)

            # zeroノードへ書き込み
            # [MAX_Connect_Number]
            zero_node_P_connect = self.node[frame - 1, 0, -1]

            count = (zero_node_P_connect != self._ignore_node_number).sum()
            self.node[frame - 1, 0, -1, count: count + F3_F1_idx.size(0)] = F3_F1_idx
            
            
            # frame-2とframeの結合でも0だった場合frame-3でも検証が行われる
            zero_match = zero_match[zero_flag]

            # 確信度を計算するためのベクトルを取得 [past idx, correct idx]
            F3_F0_idx = self.connect_inner[2][frame - 3, :, zero_match]

            # ゼロマッチングした検出が選ぶF0のidxを取得
            F3_F0_idx = F3_F0_idx.max(dim=0)[1]

            # ゼロ削除
            list_t_idx = zero_match[F3_F0_idx != 0]
            F3_F0_idx = F3_F0_idx[F3_F0_idx != 0]

            # ホワイトリスト登録
            white_list = F3_F0_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame
            # t-3 idx　既にしているため実行しなくてよい
            #white_list[:, 1] = F3_F0_idx
            # t-2 idx
            white_list[:, 2] = 0
            # t-1 idx
            white_list[:, 3] = 0
            # t idx
            white_list[:, 4] = list_t_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            # 重複削除
            F3_F0_idx = torch.unique(F3_F0_idx)

            # zeroノードへ書き込み
            # [MAX_Connect_Number]
            zero_node_P_connect = self.node[frame - 2, 0, -1]

            count = (zero_node_P_connect != self._ignore_node_number).sum()
            self.node[frame - 2, 0, -1, count: count + F3_F0_idx.size(0)] = F3_F0_idx

    def nxt_detection_error_fix(self):
        # 現在から未来へのマッチングにおいてゼロにマッチングしたもの

        for frame in range(self.node.size(0) - 3):
            # 対象frameからゼロにマッチングしたもの
            # ignore判定も同時に行っている
            # ゼロとマッチングする結合線を1(現在から未来)に追加することはないとする
            zero_match = self.node[frame, self.node[frame, :, 1, 0] == 0, 0, 0]

            # PAD削除
            PAD_flag = self.coord[frame, zero_match, 0] == self._PAD_coord
            zero_match = zero_match[~PAD_flag]

            # 確信度を計算するためのベクトルを取得 [past idx,correct idx]
            F0_F2_idx = self.connect_inner[1][frame, zero_match]

            # ゼロマッチングした検出が選ぶF1のidxを取得
            F0_F2_idx = F0_F2_idx.max(dim=-1)[1]

            # ゼロ削除,ゼロの位置は保存する
            zero_flag = F0_F2_idx == 0
            t_idx = zero_match[~zero_flag]
            F0_F2_idx = F0_F2_idx[~zero_flag]

            # ホワイトリスト登録
            white_list = F0_F2_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame + 2
            # t-3 idx
            white_list[:, 1] = self._ignore_node_number
            # t-2 idx
            white_list[:, 2] = t_idx
            # t-1 idx
            white_list[:, 3] = 0
            # t idx 既にしているため実行しなくてよい
            #white_list[:, 4] =  F2_F0_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            # 重複削除
            F0_F2_idx = torch.unique(F0_F2_idx)

            # zeroノードへ書き込み
            # [MAX_Connect_Number]
            zero_node_P_connect = self.node[frame + 1, 0, 1]

            count = (zero_node_P_connect != self._ignore_node_number).sum()
            self.node[frame + 1, 0, 1, count: count + F0_F2_idx.size(0)] = F0_F2_idx

            
            # frame-2とframeの結合でも0だった場合frame-3でも検証が行われる
            zero_match = zero_match[zero_flag]

            # 確信度を計算するためのベクトルを取得 [past idx,correct idx]
            F0_F3_idx = self.connect_inner[2][frame, zero_match]

            # ゼロマッチングした検出が選ぶF0のidxを取得
            F0_F3_idx = F0_F3_idx.max(dim=-1)[1]

            # ゼロ削除
            t_idx = zero_match[F0_F3_idx != 0]
            F0_F3_idx = F0_F3_idx[F0_F3_idx != 0]

            # ホワイトリスト登録
            white_list = F0_F3_idx[:, None].repeat(1, 5)
            # frame
            white_list[:, 0] = frame + 3
            # t-3 idx
            white_list[:, 1] = t_idx
            # t-2 idx
            white_list[:, 2] = 0
            # t-1 idx
            white_list[:, 3] = 0
            # t idx 既にしているため実行しなくてよい
            #white_list[:, 4] =  F0_F3_idx

            self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

            # 重複削除
            F0_F3_idx = torch.unique(F0_F3_idx)

            # zeroノードへ書き込み
            # [MAX_Connect_Number]
            zero_node_P_connect = self.node[frame + 2, 0, 1]

            count = (zero_node_P_connect != self._ignore_node_number).sum()
            self.node[frame + 2, 0, 1, count: count + F0_F3_idx.size(0)] = F0_F3_idx
            

        # 最後のフレームはframeとframe+2の情報のみ使える
        # 対象frameからゼロにマッチングしたもの
        # ignore判定も同時に行っている
        # ゼロとマッチングする結合線を1(現在から未来)に追加することはないとする
        zero_match = self.node[-3, self.node[-3, :, 1, 0] == 0, 0, 0]
        # PAD削除
        PAD_flag = self.coord[-3, zero_match, 0] == self._PAD_coord
        zero_match = zero_match[~PAD_flag]

        # 確信度を計算するためのベクトルを取得 [past idx,correct idx]
        F0_F2_idx = self.connect_inner[1][-1, zero_match]

        # ゼロマッチングした検出が選ぶF1のidxを取得
        F0_F2_idx = F0_F2_idx.max(dim=-1)[1]

        # ゼロ削除
        t_idx = zero_match[F0_F2_idx != 0]
        F0_F2_idx = F0_F2_idx[F0_F2_idx != 0]

        # ホワイトリスト登録
        white_list = F0_F2_idx[:, None].repeat(1, 5)
        # frame
        white_list[:, 0] = frame + 2
        # t-3 idx
        white_list[:, 1] = self._ignore_node_number
        # t-2 idx
        white_list[:, 2] = t_idx
        # t-1 idx
        white_list[:, 3] = 0
        # t idx 既にしているため実行しなくてよい
        #white_list[:, 4] =  F2_F0_idx

        self.node_white_list = torch.cat([self.node_white_list, white_list], dim=0)

        # 重複削除
        F0_F2_idx = torch.unique(F0_F2_idx)

        # zeroノードへ書き込み
        # [MAX_Connect_Number]
        zero_node_P_connect = self.node[-1, 0, 1]

        count = (zero_node_P_connect != self._ignore_node_number).sum()
        self.node[-2, 0, 1, count: count + F0_F2_idx.size(0)] = F0_F2_idx

        pass

    def apply_white_list(self, track, frame):
        # detection fixの制御 frameの次の検出を追加するためframe + 1
        white_list = self.node_white_list[self.node_white_list[:, 0] == frame + 1]

        # white listのt-1,tに一致した追跡を抽出
        white_list_connect_flag = (track[None, -2:] == white_list[:, -2:, None])
        white_list_connect_flag = torch.all(white_list_connect_flag, dim=1)
        white_list_connect_flag = torch.any(white_list_connect_flag, dim=0)

        # white_listのt-3,t-2,t-1,tに一致した追跡を抽出
        # white_listのt-3,t-2,t-1に一致した追跡も抽出
        if track[-4:].size(0) == 3:  # frame=1のときはtrackの長さが足りずt-3まで計算できない
            white_list_track_flag = track[None, -3:] == white_list[:, 2:, None]
            white_list_track_all_flag = track[None, - 3:-1] == white_list[:, 2:-1, None]
        else:  # もし4frameの計算ができたらならignoreを考慮に入れる
            white_list_track_flag = (track[None, -4:] == white_list[:, 1:, None]) | (white_list[:, 1:, None] == self._ignore_node_number)
            white_list_track_all_flag = (track[None, -4:-1] == white_list[:, 1:-1, None]) | (white_list[:, 1:-1, None] == self._ignore_node_number)

        white_list_track_flag = torch.all(white_list_track_flag, dim=1)
        white_list_track_flag = torch.any(white_list_track_flag, dim=0)

        white_list_track_all_flag = torch.all(white_list_track_all_flag, dim=1)
        white_list_track_all_flag = torch.any(white_list_track_all_flag, dim=0)

        # ホワイトリストにない0マッチはすべて途中出現や、途中消失
        # 途中消失だが検出してしまった場合のフラグ
        disappearance_flag = torch.any(track[:-2] != 0, dim=0) & (track[-2] == 0) & (track[-1] != 0)

        # 完全一致のものは消さずに、結合のみ一致したものは削除する
        #True is delete
        delete_track_flag1 = white_list_connect_flag & ~white_list_track_flag

        # 途中消失は削除
        #True is delete
        delete_track_flag2 = disappearance_flag & ~white_list_connect_flag

        # ホワイトリストに登録されている追跡は他の検出に関連付けしてはいけない
        delete_track_flag3 = white_list_track_all_flag & ~white_list_connect_flag

        delete_track_flag = delete_track_flag1 | delete_track_flag2 | delete_track_flag3

        return delete_track_flag

    def delete_multi_branch(self, track):
        frame = track.size(0) - 1

        # [frame,num] => [num,frame]
        track = track.transpose(0, 1)

        # t-1でかぶりがある物を抽出,indexの大きさは元のサイズと同じ
        # 複数検知の物は同じ数字がある
        _, index = torch.unique(track[:, -3], return_inverse=True)
        # flagには重複している個数が入っている
        flag = (index[:, None] == index[None, :]).sum(dim=-1)
        # 2以上のものはTrue
        flag = flag > 1

        overlap = track[flag, -3:]

        # tの検出が同じ物があるといけないためそれを除去する
        flag = torch.all(overlap[:, None] == overlap[None, :], dim=-1)
        # flagには重複している個数が入っている
        flag = flag.sum(dim=-1)
        # 2以上のものはTrue
        flag = flag > 1
        # overlapはt-1で重複があり、かつtには重複がないものになる
        overlap = overlap[~flag]

        # t-1とt+1でかぶりがある物を抽出
        flag = torch.all(overlap[:, None, [0, 2]] == overlap[None, :, [0, 2]], dim=-1)

        # flagには重複している個数が入っている
        flag = flag.sum(dim=-1)
        # 2以上のものはTrue
        flag = flag > 1

        # overlapはt-1で重複があり、かつt+1で重複して、かつtには重複がないものになる
        overlap = overlap[flag]

        # もしoverlapがないなら終了
        if overlap.size(0) == 0:
            return track.new_zeros([track.size(0)], dtype=bool)

        # t-1とt+1の種類を取得する
        # t-1とt+1でかぶりがある物を抽出
        flag = torch.all(overlap[:, None, [0, 2]] == overlap[None, :, [0, 2]], dim=-1)

        # かぶりがあるもので最初の物の値を抽出
        # maxは最大値の物で一番idxが低いidxを取得することを利用
        flag = flag.max(dim=-1)[1]

        # idxからone hot vectorを作成
        flag = one_hot_changer(flag, overlap.shape[0], dim=-1, bool_=True)

        # 最初に出現するものはTrueが重なりそれ以降の物はdim=0軸ですべてFalseになる
        flag = torch.any(flag, dim=0)

        # overlap[flag,[0,2]]はできないため仕方なく
        overlap_kinds = overlap[flag][:, [0, 2]]

        # 種類ごとに分ける
        split_overlap = [overlap[torch.all(
            overlap[:, [0, 2]] == kind, dim=-1)] for kind in overlap_kinds]

        split_overlap = torch.nn.utils.rnn.pad_sequence(
            split_overlap, batch_first=True)

        # 確信度の合計が一番高いものを選ぶ
        past_vector = self.connect_vector[0][frame - 2, self.coord[frame - 2, :, 0] != -1]
        pre_vector = self.connect_vector[0][frame -
                                            1, self.coord[frame - 1, :, 0] != -1]
        nxt_vector = self.connect_vector[0][frame,
                                            self.coord[frame, :, 0] != -1]

        # 0frame と 3frame の類似度
        past_pre_similar = (past_vector[:, None]
                            * pre_vector[None, :]).sum(dim=-1)
        pre_nxt_similar = (pre_vector[:, None] * nxt_vector[None, :]).sum(dim=-1)

        connect_value = past_pre_similar[split_overlap[:, :, 0], split_overlap[:, :, 1]] + pre_nxt_similar[split_overlap[:, :, 1], split_overlap[:, :, 2]]

        # PADの値を0にする
        connect_value[torch.all(split_overlap == 0, dim=-1)] = 0.

        # 確信度が一番高い物のidxをとる
        connect_idx = connect_value.max(dim=-1)[1]

        # 確信度が一番高い物のidxの追跡結果を取り出す
        split_overlap = torch.gather(
            split_overlap, dim=1, index=connect_idx[:, None, None].expand(-1, -1, 3))
        split_overlap = split_overlap.squeeze(1)

        # 確信度が一番高い物と一致しないものを抽出
        flag = torch.all(overlap[:, None] == split_overlap[None, :], dim=-1)
        flag = torch.any(flag, dim=-1)
        delete_track = overlap[~flag]

        # 削除対象と一致するものをTrueにしそれを消す
        flag = torch.all(track[:, None, -3:] == delete_track[None, :], dim=-1)
        flag = torch.any(flag, dim=-1)
        return flag

    def delete_same_track(self, track):
        # 最後のtrack idxは必ず1つは保持する必要がある
        # そのため最後のフレームでグループわけを行い、そこから削除するtrackを決定していく
        # 削除アルゴリズム
        # step1:
        # 一番同じようなtrackをもつ中心trackを決める。判定は厳しめで
        # step2:
        # その中から1つ最も確信度が高いものを選ぶ
        # step3:
        # 選んだもの以外を削除する
        # 一度の操作ではすべてのsame trackを削除できない仕組み

        last_idx = torch.unique(track[-1, :])
        # 0は削除
        last_idx = last_idx[last_idx != 0]

        delete_flag = track.new_zeros([track.size(1)], dtype=bool)

        for idx in last_idx:
            select_flag = track[-1, :] == idx
            select_track = track[:, select_flag]
            # [frame,num,2(x,y)]
            track_coord = torch.gather(self.coord[: select_track.size(
                0)], dim=1, index=select_track[:, :, None].expand(-1, -1, 2))

            # NoneやPADはマッチしないようにする。ただしNone同士はマッチする
            track_coord[track_coord < 0] = -self.inf

            # 距離計算
            distance = (track_coord[:, :, None] - track_coord[:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

            # マッチしたかを検証
            distance = distance <= self.correct_range
            # 現在frameでマッチしたか割合を計算
            distance = distance.float().mean(dim=0)
            # マッチした割合が一定以上であるか検証
            distance = distance >= self.accuracy_rate

            # マッチした個数が一番多いtrackを抽出
            match_max_value, match_max_idx = torch.max(
                distance.sum(dim=-1), dim=0)

            if match_max_value == 1:
                continue

            # 最大のtrackを中心にマッチしたtrackを選ぶ
            match_flag = distance[match_max_idx]
            match_track = select_track[:, match_flag]

            # gatherのための整形
            match_track = match_track[:, :,
                                      None].expand(-1, -1, self.connect_vector[0].shape[-1])
            # [frame,num,dim]
            match_vector = torch.gather(
                self.connect_vector[0], dim=1, index=match_track)

            # 確信度計算
            connect_value = (match_vector[:-1] * match_vector[1:]).sum(dim=-1)
            connect_value = connect_value.mean(dim=0)  # frame方向に

            # 最大のidxを取得
            connect_max_idx = torch.argmax(connect_value)

            # 削除しないtrackはFalseに
            match_flag[match_flag.cumsum(dim=0) == connect_max_idx + 1] = False
            delete_track = select_track[:, match_flag]

            # 削除するtrackを検索
            delete_flag_temp = torch.all(
                track[:, :, None] == delete_track[:, None], dim=0)
            delete_flag_temp = torch.any(delete_flag_temp, dim=-1)

            delete_flag = delete_flag | delete_flag_temp

        return delete_flag

    def delete_all_same_track(self, track, distance=None):
        # 探索範囲はすべて
        # 削除アルゴリズム
        # step1:
        # 一番同じようなtrackをもつ中心trackを決める。判定は厳しめで
        # step2:
        # その中から1つ最も確信度が高いものを選ぶ
        # step3:
        # 選んだもの以外を削除する
        # 一度の操作ではすべてのsame trackを削除できない仕組み
        if distance is None:
            # [frame,num,2(x,y)]
            track_coord = torch.gather(self.coord[: track.size(0)], dim=1, index=track[:, :, None].expand(-1, -1, 2))

            # NoneやPADはマッチしないようにする。ただしNone同士はマッチする
            track_coord[track_coord < 0] = -self.inf

            # 距離計算
            distance = (track_coord[:, :, None] - track_coord[:, None, :]) ** 2
            distance = torch.sqrt(distance.sum(dim=-1))

            # マッチしたかを検証
            distance = distance <= self.correct_range
            # 現在frameでマッチしたか割合を計算
            distance = distance.float().mean(dim=0)
            # マッチした割合が一定以上であるか検証
            distance = distance >= self.accuracy_rate

        # マッチした個数が一番多いtrackを抽出
        match_max_value, match_max_idx = torch.max(distance.sum(dim=-1), dim=0)

        if match_max_value == 1:
            return track.new_zeros([track.size(1)], dtype=bool), distance

        # 最大のtrackを中心にマッチしたtrackを選ぶ
        match_flag = distance[match_max_idx]
        match_track = track[:, match_flag]

        # gatherのための整形
        match_track = match_track[:, :, None].expand(-1, -1, self.connect_vector[0].shape[-1])
        # [frame,num,dim]
        match_vector = torch.gather(self.connect_vector[0], dim=1, index=match_track)

        # 確信度計算
        connect_value = (match_vector[:-1] * match_vector[1:]).sum(dim=-1)
        connect_value = connect_value.mean(dim=0)  # frame方向に

        # 最大のidxを取得
        connect_max_idx = torch.argmax(connect_value)

        # 削除しないtrackはFalseに
        match_flag[match_flag.cumsum(dim=0) == connect_max_idx + 1] = False
        delete_track = track[:, match_flag]

        # 削除するtrackを検索
        delete_flag = torch.all(track[:, :, None] == delete_track[:, None], dim=0)
        delete_flag = torch.any(delete_flag, dim=-1)

        return delete_flag, distance

    def delete_move_vector_error(self, track_idx):
        """移動ベクトルの異常を削除する関数、move limit以上になったら削除する

        Args:
            track_idx (tensor[frame,num]): tracking coordinate

        Returns:
            tensor[num]: delete flag.True is delete track
        """        

        track_coordinate = torch.gather(self.coord[: track_idx.size(0)], dim=1, index=track_idx[:, :, None].expand(-1, -1, 2))
        #[frame,num,2(x,y)]

        #NoneをNoneになる前の座標にする
        None_flag = torch.any(track_coordinate == -2,dim=-1)

        #通常のidx
        arange = torch.arange(None_flag.shape[0])[:, None].expand(-1, None_flag.shape[1])
        
        #増加フラグを作成
        diff_zero_flag = None_flag.int().diff(n=1, dim=0)
        increase = torch.cat([diff_zero_flag.new_zeros([1, diff_zero_flag.shape[1]]), diff_zero_flag < 0])

        #NoneのときだけNoneの一番初めのidxになるように作成
        None_arange_idx = (~None_flag).cumsum(dim=0)-1
        diff = (arange-None_arange_idx)*increase
        diff = (2*diff-diff.cumsum(dim=0))*increase
        None_arange_idx = (~None_flag*1.0 + diff).cumsum(dim=0)-1
        None_arange_idx = None_arange_idx.long()
        None_arange_idx[~None_flag] = arange[~None_flag]

        #最初にゼロがあるものを対処
        #-1は最初にゼロがあるもの
        first_zero_flag = None_arange_idx < 0
        #-1以外で最小の値を取得
        track_idx_min = torch.min(None_arange_idx[~first_zero_flag])
        track_idx_min = torch.full_like(None_arange_idx, track_idx_min)
        None_arange_idx[first_zero_flag] = track_idx_min[first_zero_flag]

        #gatherのために整形
        None_arange_idx = None_arange_idx[:, :, None].expand(-1, -1, 2)

        track_coordinate = torch.gather(track_coordinate, dim=0, index=None_arange_idx)

        #移動ベクトルの計算
        move_vector = torch.diff(track_coordinate, n=1, dim=0)

        #移動ベクトルの大きさを計算
        #torch.normを使うと処理が遅くなる
        move_norm = (move_vector ** 2).sum(dim=-1)
        move_norm = torch.sqrt(move_norm)

        #移動ベクトルの大きさが閾値以上ならerrorと判断 [num]
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

    def Debag_PAD_not_have_connect(self):
        # nodeのPADを抽出する [PAD num,3,MAX_Connect_Number]
        PAD_flag = self.coord[:, :, 0] == self._PAD_coord
        PAD_node = self.node[PAD_flag]

        # -1,1の結合結果を抽出
        PAD_node = PAD_node[:, [-1, 1]]
        if ~torch.all(PAD_node == -1):
            print("PADの結合が無視されていません")

    def Debag_get_Combination(self):
        # 現在でどれくらいの組み合わせ(track数)になるのか計算する関数

        count_node = self.node.new_ones(self.node.shape[:2])
        # PADのみ0に None tokenは1のまま
        count_node[self.coord[:, :, 0] == -1] = 0

        for i in range(1, count_node.size(0)):  # 1からlast frameまで
            # 現在から過去へのマッチングを取得
            match = self.get_connect_node(i, mode="P")
            # Current idx取得
            C_idx = self.node[i, :, 0, 0]
            # Previous count取得
            P_count = count_node[i - 1]

            # マッチのそれ以外のidxの部分をP_countに置き換える
            match[:, 1] = P_count[match[:, 1]]

            # 現在へのcountを計算
            count = (C_idx[:, None] == match[None, :, 0]) * match[None, :, 1]
            count = count.sum(dim=-1)

            # 現在のcountとかける(PAD対策)
            count_node[i] = count * count_node[i]

        Comb_total = count_node[-1].sum()
        print(Comb_total)

        pass


def one_hot_changer(tensor, vector_dim, dim=-1, bool_=False):
    """index tensorをone hot vectorに変換する関数

    Args:
        tensor (torch.tensor,dtype=torch.long): index tensor
        vector_dim (int): one hot vectorの次元。index tensorの最大値より大きい値でなくてはならない
        dim (int, optional): one hot vectorをどこの次元に組み込むか. Defaults to -1.
        bool_ (bool, optional): Trueにするとbool型になる。Falseの場合はtorch.float型. Defaults to False.

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
        raise TypeError("入力テンソルがtorch.long型ではありません")
    if tensor.max() >= vector_dim:
        raise ValueError("入力テンソルのindex番号がvector_dimより大きくなっています")

    # one hot vector用単位行列
    one_hot = torch.eye(vector_dim, dtype=data_type, device=tensor.device)
    vector = one_hot[tensor, ]

    # one hot vectorの次元変更
    dim_change_list = list(range(tensor.dim()))
    # もし-1ならそのまま出力
    if dim == -1:
        return vector
    # もしdimがマイナスならスライス表記と同じ性質にする
    if dim < 0:
        dim += 1  # omsertは-1が最後から一つ手前

    dim_change_list.insert(dim, tensor.dim())
    vector = vector.permute(dim_change_list)
    return vector
