#coding: utf-8
#----- Standard Library -----#
import os
import random
import sys
import math

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
import cv2

#----- Module -----#
from utils.dataset import Konmo_Loader, Konmo_Video_Loader
import utils.transforms_seg as tf
from utils.evaluation import IoU
from models.Unet_2D import UNet, UNet_Domain

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([
                                tf.Padding_div(32)])

    test_dataset = Konmo_Loader(root_dir=cfg.dataset.root_dir, mode="test", transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    
    return test_loader


############## test function ##############
@torch.no_grad()
def test(model, test_loader, device, cfg):
    model.eval()

    # init setting
    sum_loss = 0
    iou_calc = IoU(2)

    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
        
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.long().cuda(device, non_blocking=True)

        output = model(inputs)
        mask = output.max(dim=1)[1]
        mask = torch.stack([mask, mask, mask], dim=1)
        images = torch.cat([inputs, mask], dim=0)

        # torchvision.utils.save_image(images,
        #                             f"result{batch_idx:02}.png",
        #                             normalize=True)

        iou_calc.update(output, targets)
    

    acc = iou_calc()


    return acc


############## test function ##############
@torch.no_grad()
def test_video(model, test_loader, device, cfg):
    model.eval()
    path = "/mnt/kamiya/dataset/konmo/P1270204/P1270216.JPG"
    inputs = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    inputs = torch.from_numpy(inputs).float() / 255.
    inputs, _ = test_loader.dataset.transform(inputs, inputs)
    inputs = inputs[None].cuda(device, non_blocking=True)

    output = model(inputs)
    mask = output.max(dim=1)[1]
    mask = torch.stack([mask, mask, mask], dim=1)
    images = torch.cat([inputs, mask], dim=0)

    torchvision.utils.save_image(images,
                                f"result_video.png",
                                normalize=True)


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")

    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test.txt"

    with open(PATH, mode='w') as f:
        f.write("")
    
    # data load
    test_loader = dataload(cfg)

    # def model
    model = UNet(in_channels=3,
                n_classes=2).cuda(device)
    
    # model = UNet_Domain(in_channels=3,
    #             n_classes=2).cuda(device)


    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    torch.backends.cudnn.benchmark = True

    iou = test(model, test_loader, device, cfg)

    with open(PATH, mode='a') as f:
        for k in iou.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in iou.values():
            f.write(f"{v}\t")
    
    print("IoU Evalution")
    for k, v in iou.items():
        print(f"{k}".ljust(20) + f":{v}")

    test_video(model, test_loader, device, cfg)

############## main ##############
if __name__ == '__main__':
    main()
