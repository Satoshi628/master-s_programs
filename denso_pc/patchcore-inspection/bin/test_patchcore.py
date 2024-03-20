import os
import time

import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.patchcore_fast
import patchcore.sampler
import patchcore.utils


device = "cuda:0"
layers_to_extract_from = ["layer2", "layer3"]
input_shape = (3, 224, 224)
backbone = patchcore.backbones.load("wideresnet101")
pretrain_embed_dimension = 1024
target_embed_dimension = 1024
patchsize = 3
anomaly_scorer_num_nn = 1

nn_method = patchcore.common.FaissNN(False, 8)
sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(0.1, device)

model = patchcore.patchcore_fast.PatchCore(device)
model.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract_from,
    device=device,
    input_shape=input_shape,
    pretrain_embed_dimension=pretrain_embed_dimension,
    target_embed_dimension=target_embed_dimension,
    patchsize=patchsize,
    featuresampler=sampler,
    anomaly_scorer_num_nn=anomaly_scorer_num_nn,
    nn_method=nn_method,
)

dummy_input = torch.rand([1,3,224,224], device=device)
output = model._embed(dummy_input)
print("outputs", output.shape)