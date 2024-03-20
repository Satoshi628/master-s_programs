#coding: utf-8
#----- Standard Library -----#
#None

#----- Public Package -----#
import numpy as np
import torch
import torch.nn as nn

#----- Module -----#
#None


def get_movelimit(dataset, factor=0.9):
    all_tracks = dataset.get_track()

    move_max_list = [[], [], []]
    cash_move_append = [move_list.append for move_list in move_max_list]

    for track in all_tracks:
        # frame sort
        sort_idx = track[:, 0].sort()[1]
        track = track[sort_idx]
        zero_point = track[track[:, 0] == 0]
        aa = torch.unique(zero_point[:, 1], return_counts=True)

        #frame,id,x,y
        unique_id = torch.unique(track[:, 1])
        for target_id in unique_id:
            for i in range(1, 4):
                target_track = track[track[:, 1] == target_id][:, [2, 3]]
                move = target_track[:-i] - target_track[i:]
                move = torch.sqrt((move ** 2).sum(dim=-1))

                if move.shape[0] != 0:
                    cash_move_append[i - 1](move.max())

    del cash_move_append

    move_limit = [factor * max(move) for move in move_max_list]
    return move_limit
