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

def show_anomaly(images, anomaly, file_names, save_path):
    anomaly_max = anomaly.max()
    anomaly_min = anomaly.min()
    
    print("show anomaly maps")
    for img, ano, name in zip(tqdm(images), anomaly, file_names):
        img, ano = img[None], ano[None]
        img, ano = torch.from_numpy(img), torch.from_numpy(ano)

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img*std[None, :, None, None] + mean[None, :, None, None]

        anomaly = (anomaly-anomaly_min) / (anomaly_max - anomaly_min + 1e-8)
    
        #convert color jet
        ano = ano.numpy() *255
        ano = ano.astype(np.uint8)
        ano = np.stack([cv2.applyColorMap(a[0], cv2.COLORMAP_JET) for a in ano])
        ano = torch.from_numpy(ano).float() /255.
        ano = ano.permute(0,3,1,2)[:,[2,1,0]]

        rate = 0.7
        ano = img*rate + ano*(1-rate)

        groups = torch.cat([img, ano], dim=0)
        name = os.path.splitext(os.path.basename(name))[0]
        torchvision.utils.save_image(groups,
                                    f"{save_path}/anomaly_{name}.png",
                                    #nrow=1,
                                    normalize=True)





