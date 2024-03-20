#coding: utf-8
#----- Standard Library -----#
import os
import random
import sys
import math
import itertools

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
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Konmo_Loader, Konmo_Video_Loader
import utils.transforms_seg as tf
from utils.evaluation import IoU
from utils.loss import IoULoss
from models.Unet_2D import UNet_Domain


############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.Resize(size=512),
                                tf.RandomCrop(size=(512, 512)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    val_transform = tf.Compose([tf.Resize(size=512),
                                tf.Padding_div(32)])
    train_dataset = Konmo_Loader(root_dir=cfg.dataset.root_dir, mode="train", transform=train_transform)
    targets_dataset = Konmo_Video_Loader(root_dir=cfg.dataset.root_dir, transform=train_transform)

    val_dataset = Konmo_Loader(root_dir=cfg.dataset.root_dir, mode="val", transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        # num_workers=2,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=2,
        drop_last=False)
    targets_loader = torch.utils.data.DataLoader(
        targets_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=2,
        drop_last=False)
    
    targets_loader = itertools.cycle(targets_loader)
    return train_loader, targets_loader, val_loader

############## train function ##############
def train(model, train_loader, targets_loader, criterion, criterion_da, optimizer, device):
    model.train()

    # init setting
    sum_loss = 0
    
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.long().cuda(device, non_blocking=True)
        domain_data = next(targets_loader).cuda(device, non_blocking=True)
        
        inputs = torch.cat([inputs, domain_data], dim=0)
        output, domain = model(inputs)
        
        domain_label = torch.tensor([0 for _ in range(targets.shape[0])] + [1 for _ in range(domain_data.shape[0])], dtype=torch.long, device=device)
        loss_da = criterion_da(domain, domain_label)
        loss = criterion(output[: targets.shape[0]], targets)
        loss = loss + loss_da*0.01

        optimizer.zero_grad()
        loss.backward()

        sum_loss += loss.item()

        del loss    # free memory
        optimizer.step()
    
    return sum_loss / (batch_idx + 1)

############## validation function ##############
@torch.no_grad()
def val(model, val_loader, criterion, device, cfg):
    model.eval()

    # init setting
    sum_loss = 0
    iou_calc = IoU(2)

    for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
        
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.long().cuda(device, non_blocking=True)

        output, _ = model(inputs)
        loss = criterion(output, targets)
        sum_loss += loss.item()
        
        iou_calc.update(output, targets)

    acc = iou_calc()


    return sum_loss / batch_idx, acc

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print("Epoch".ljust(20) + f":{cfg.parameter.epoch}")
    print("Batch size".ljust(20) + f":{cfg.parameter.batch_size}")
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    
    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        text = "epoch\t" + "\t".join([k for k in IoU(2).keys()]) + "\n"
        # text = "epoch\t" + "\t".join([k for k in Object_Detection(right_range=10.).keys()]) + "\n"
        f.write(text)
    
    # data load
    train_loader, targets_loader, val_loader = dataload(cfg)


    # def model
    model = UNet_Domain(in_channels=3,
                n_classes=2).cuda(device)
    
    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    criterion_da = nn.CrossEntropyLoss()
    criterion = IoULoss(2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    torch.backends.cudnn.benchmark = True

    best_mIoU = 0.0
    best_loss = 99999.
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, targets_loader, criterion, criterion_da, optimizer, device)
        # validation
        val_loss, val_acc = val(model, val_loader, criterion, device, cfg)

        mIoU = val_acc["mIoU"]
        # mIoU = val_acc["F1_Score"]

        text = f"Epoch{epoch + 1:3d}/{cfg.parameter.epoch:3d} TrainLoss={train_loss:.5f} ValLoss={val_loss:.5f}" \
            + f" Val mIoU:{mIoU:.2%}"
        print(text)

        ##### Writing result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            text = f"{int(epoch+1)}\t" + "\t".join([f"{v:.4f}" for v in val_acc.values()]) + "\n"
            f.write(text)
        

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

    print("Best mIoU:{:.2%}".format(best_mIoU))

############## main ##############
if __name__ == '__main__':
    main()
