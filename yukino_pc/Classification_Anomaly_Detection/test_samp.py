#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import time
#----- 専用ライブラリ -----#
from skopt.sampler import Hammersly
from sklearn import cluster
import numpy as np
from numpy.linalg import norm
import torch


data = np.random.normal(size=[1000,128])
# start_time = time.time()
# ms = cluster.MeanShift(seeds=data)
# ms.fit(data)
# end_time = time.time()
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# print(cluster_centers.shape)
