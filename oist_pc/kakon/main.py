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

#----- Module -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.dataset import Zenigoke_Loader, collate_delete_PAD
import utils.transforms as tf
from utils.evaluation import Object_Detection, Calc_AP
from models.Unet_3D import UNet_3D
from models.Unet_2D import UNet_2D



############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    val_transform = tf.Compose([])
    train_dataset = Zenigoke_Loader(root_dir=cfg.dataset.root_dir, mode="train", split=cfg.dataset.split, length=16, delay=cfg.parameter.delay, transform=train_transform)
    val_dataset = Zenigoke_Loader(root_dir=cfg.dataset.root_dir, mode="val", split=cfg.dataset.split, length=16, delay=cfg.parameter.delay, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        # num_workers=2,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        # num_workers=2,
        drop_last=False)
    
    return train_loader, val_loader

############## train function ##############
def train(model, train_loader, criterion, optimizer, device):
    model.train()

    # init setting
    sum_loss = 0

    for batch_idx, (inputs, targets, point) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
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
    
    return sum_loss / (batch_idx + 1)

############## validation function ##############
@torch.no_grad()
def val(model, val_loader, criterion, device, cfg):
    model.eval()

    # init setting
    sum_loss = 0
    
    # detection_calc = Object_Detection(right_range=10)
    calc_ap = Calc_AP(right_range=[10., 15.])

    for batch_idx, (inputs, targets, point) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
        #image: input.size() => [batch,channel,length,H,W]
        #EP map: targets.size() => [batch,length,H,W]
        #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
        
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)

        output, coord = model.get_coord(inputs)
        loss = criterion(output, targets)
        sum_loss += loss.item()
        
        # detection_calc.update([c[:, [0, 1]] for c in coord], point.flatten(0, 1)[:, :, -2:])
        calc_ap.update(output, point[0])

    # acc = detection_calc()
    acc = calc_ap()
    # print("all object accuracy")
    # print(acc)


    return sum_loss / batch_idx, acc

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print("Epoch".ljust(20) + f":{cfg.parameter.epoch}")
    print("Batch size".ljust(20) + f":{cfg.parameter.batch_size}")
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    print("padding mode".ljust(20) + f":{cfg.parameter.padding}")
    
    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        text = "epoch\t" + "\t".join([k for k in Calc_AP(right_range=[10., 15.]).keys()]) + "\n"
        # text = "epoch\t" + "\t".join([k for k in Object_Detection(right_range=10.).keys()]) + "\n"
        f.write(text)
    
    # data load
    train_loader, val_loader = dataload(cfg)


    # def model
    if cfg.parameter.use_3D:
        model = UNet_3D(in_channels=3,
                        n_classes=1,
                        channel=16,
                        noise_strength=cfg.parameter.noise_strength,
                        delay_upsample=int(math.log2(cfg.parameter.delay)),
                        padding_mode=cfg.parameter.padding).cuda(device)
    else:
        model = UNet_2D(in_channels=3,
                    n_classes=1,
                    noise_strength=cfg.parameter.noise_strength,
                    delay_upsample=int(math.log2(cfg.parameter.delay)),
                    padding_mode=cfg.parameter.padding).cuda(device)
    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch


    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    torch.backends.cudnn.benchmark = True

    best_mAP = 0.0
    best_loss = 99999.
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        val_loss, val_acc = val(model, val_loader, criterion, device, cfg)

        mAP = val_acc["mAP"]
        # mAP = val_acc["F1_Score"]

        text = f"Epoch{epoch + 1:3d}/{cfg.parameter.epoch:3d} TrainLoss={train_loss:.5f} ValLoss={val_loss:.5f}" \
            + f" Val mAP:{mAP:.2%}"
        print(text)

        ##### Writing result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            text = f"{int(epoch+1)}\t" + "\t".join([f"{v:.4f}" for v in val_acc.values()]) + "\n"
            f.write(text)
        

        if mAP > best_mAP:
            best_mAP = mAP
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)

    print("Best mAP:{:.2%}".format(best_mAP))

############## main ##############
if __name__ == '__main__':
    main()
