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

#----- Module -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.loss import FPFN_MSE_Loss
from utils.dataset import OIST_Loader
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.MPM_to_track import MPM_to_Track
from utils.evaluation import Object_Tracking
from models.Unet_3D import UNet_3D



############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    val_transform = tf.Compose([])
    train_dataset = OIST_Loader(root_dir="/mnt/kamiya/dataset/NC1032", mode="train", length=16, delay=2, transform=train_transform)
    val_dataset = OIST_Loader(root_dir="/mnt/kamiya/dataset/NC1032", mode="val", length=16, delay=2, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return train_loader, val_loader

############## train function ##############
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    # Nondeterminism
    torch.backends.cudnn.deterministic = False

    # init setting
    sum_loss = 0


    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        #image: input.size() => [batch,channel,length,H,W]
        #EP map: targets.size() => [batch,length,H,W]
        #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]

        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        output = model(inputs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()

        sum_loss += loss.item()

        del loss    # free memory
        optimizer.step()
    
    # Determinism
    torch.backends.cudnn.deterministic = True
    return sum_loss / (batch_idx + 1)

############## validation function ##############
def val(model, val_loader, criterion, Track_transfor, Tracking_Eva, device):
    model.eval()
    coord_F = None
    # Determinism
    torch.backends.cudnn.deterministic = True

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            #EP map: targets.size() => [batch,length,H,W]
            #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
            
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            output = model(inputs)
            loss = criterion(output, targets)

            Track_transfor.update(vector, coord)
    
    Track_transfor()
    Tracking_Eva.update(Track_transfor.get_track(), val_loader.dataset.CP_data)
    acc = Tracking_Eva()

    Track_transfor.reset()

    # Nondeterminism
    torch.backends.cudnn.deterministic = False
    return acc

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
        f.write("epoch\tPrecision\tRecall\tF1 Score\tID Switch\n")
    
    # data load
    train_loader, val_loader = dataload(cfg)

    # def model
    model = UNet_3D(in_channels=1,
                    n_classes=1,
                    noise_strength=cfg.parameter.noise_strength,
                    delay_upsample=1).cuda(device)

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # def tracking algorithm, accuracy function
    Track_transfor = MPM_to_Track()
    tracking_fnc = Object_Tracking()

    #criterion = nn.MSELoss()
    criterion = FPFN_MSE_Loss(FP_weight=cfg.parameter.FP_weight, FN_weight=cfg.parameter.FN_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    torch.backends.cudnn.benchmark = True

    best_loss = 99999.
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        #if epoch >= 10:
        #    Tracking_acc = val(model, val_loader, criterion, Track_transfor, tracking_fnc, device)
        #else:
        #    Tracking_acc = {'mota': 0.0, 'idp': 0.0, 'idr': 0.0, 'idf1': 0.0, 'num_switches': 0}
        
        
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f}".format(
            epoch + 1,
            cfg.parameter.epoch,
            train_loss)

        print(result_text)

        ##### Writing result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        
        if train_loss < best_loss:
            best_loss = train_loss
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)
        


############## main ##############
if __name__ == '__main__':
    main()
