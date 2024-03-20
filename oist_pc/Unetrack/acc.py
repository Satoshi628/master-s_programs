#coding: utf-8
#----- Standard Library -----#
import os
import random
import sys
import time

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

#----- Module -----#
from utils.dataset import OISTLoader, collate_delete_PAD
from utils.guided_backprop import GuidedBackprop
import utils.transforms as tf
from models.Unet_3D import UNet_3D, UNet_3D_FA

transform = tf.Compose([])
dataset = OISTLoader(root_dir="/mnt/kamiya/dataset/OIST/SimDensity_mov",
                        mode="test",
                        Staning="GFP",
                        split=0,
                        length=16,
                        transform=transform)

loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        drop_last=False)


device = torch.device('cuda:0')
model = UNet_3D(in_channels=1, n_classes=1, noise_strength=0.4).cuda(device)
model_path = "/mnt/kamiya/code/Unetrack/outputs/no_assign/result/model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()


pool = nn.AvgPool2d(11, stride=1, padding=(11 - 1) // 2)

GBP = GuidedBackprop(model)

frame_diff = 1
right_range = 10.

all_count = 0
correct_count = 0

for batch_idx, (inputs, targets, point) in enumerate(tqdm(loader, leave=False)):
	#input.size() => [batch,channel,length,H,W]
	#targets.size() => [batch,length,H,W]
	#point.size() => [batch,length,num,4(frame,id,x,y)]
	#GPUモード高速化。入力、labelはfloat32
	inputs = inputs.cuda(device, non_blocking=True)
	point = point.cuda(device, non_blocking=True)
	point = point.long()
	
	for frame in range(1, point.shape[1] - 1):
		print(frame)
		for track_num in range(point.shape[2]):
			start = time.time()
			print(track_num)
			
			#grad = GBP.generate_gradients(inputs, point[0, 1, 24, 2:], frame)
			grad = GBP.generate_gradients(inputs, point[0, frame, track_num, 2:], frame)

			
			grad = grad.max(dim=0)[0]
			grad = pool(grad)
			
			max_idx = torch.nonzero(grad.flatten(1).max(dim=1)[0][:, None, None] == grad)[:, 1:]

			label = point[point[:, :, :, 1] == point[0, frame, track_num, 1]][:, [-1, -2]].cpu()
			if label.shape[0] != 16:
				continue

			frame_predict_coord = max_idx[[frame - frame_diff, frame + frame_diff]]
			frame_label_coord = label[[frame - frame_diff, frame + frame_diff]]
			frame_predict_coord = frame_predict_coord
			distance = (frame_predict_coord - frame_label_coord) ** 2.
			distance = torch.sqrt(distance.sum(dim=-1))
			
			all_count += (distance <= right_range).sum()
			correct_count += distance.shape[0]
			

			end = time.time()
			print(f"time:{end-start:.2f}")

	print(correct_count / all_count)