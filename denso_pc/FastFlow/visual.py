import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2

def show_quant_indices(indices, quant_num=512):
    colors = torch.rand([quant_num, 3])
    for i, index in enumerate(indices):
        imgs = colors[index].permute(0,3,1,2)
        torchvision.utils.save_image(imgs,
                                f"quant_index_{i}.png",
                                normalize=True)

def show_anomaly(images, labels, anomaly, save_path, limit):
    images, labels, anomaly = images.cpu(), labels.cpu(), anomaly.cpu()
    labels = labels.expand(-1,3,-1,-1)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    images = images*std[None, :, None, None] + mean[None, :, None, None]

    anomaly = (anomaly-limit[0]) / (limit[1] - limit[0] + 1e-8)

    #convert color jet
    anomaly = anomaly.detach().cpu().numpy() *255
    anomaly = anomaly.astype(np.uint8)
    anomaly = np.stack([cv2.applyColorMap(ano[0], cv2.COLORMAP_JET) for ano in anomaly])
    anomaly = torch.from_numpy(anomaly).float() /255.
    anomaly = anomaly.permute(0,3,1,2)[:,[2,1,0]]

    rate = 0.7
    anomaly = images*rate + anomaly*(1-rate)

    groups = torch.cat([images, labels, anomaly], dim=0)
    torchvision.utils.save_image(groups,
                                save_path,
                                nrow=len(images),
                                normalize=True)




