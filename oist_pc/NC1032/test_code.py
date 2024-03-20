#coding: utf-8
#----- Standard Library -----#
import os
import random
import sys

#----- Public Package -----#
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import lap
#----- Module -----#
pass


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    print(cost)
    print(x)
    print(y)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


# cost = np.random.rand(5, 6)
# print(cost)
# matches, unmatched_a, unmatched_b = linear_assignment(cost, 0.1)
# print(matches)
# print(unmatched_a)
# print(unmatched_b)

idx = list(range(0,100,15))

print(idx)
h, w = [760, 1088]
import math
ph = math.ceil(h / 32) * 32
pw = math.ceil(w / 32) * 32
print(ph)
print(pw)