#coding: utf-8
#----- 標準ライブラリ -----#
import math

#----- 専用ライブラリ -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

#----- 自作モジュール -----#
#None

channel = 32

model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            decoder_channels=[channel * 16, channel * 8, channel * 4, channel * 2, channel],
            in_channels=1,        # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,              # model output channels (number of classes in your dataset)
        ).cuda()

print(model)

inputs = torch.rand([1, 1, 128, 128]).cuda()
feature_map = model.encoder(inputs)
outputs = model.decoder(*feature_map)
classes = model.segmentation_head(outputs)
for f in feature_map:
    print("feature map:",f.shape)
print("outputs:",outputs.shape)
print("classes:",classes.shape)