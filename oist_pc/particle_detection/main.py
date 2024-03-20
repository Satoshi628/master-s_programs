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
import celldetection as cd

#----- Module -----#
from utils.scheduler import CosineAnnealingWarmupRestarts
from utils.loss import weighted_MSELoss
from utils.dataset import OIST_Loader_Train30, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_detection_copy, MPM_to_Track_multi
from utils.evaluation import Object_Detection, Object_Tracking
from models.Unet_3D import UNet_3D



############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    train_transform = tf.Compose([tf.RandomCrop(size=(256, 256)),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.RandomVerticalFlip(p=0.5),
                                ])

    val_transform = tf.Compose([])
    train_dataset = OIST_Loader_Train30(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="train", split=cfg.OIST.split, length=16, delay=1, transform=train_transform)
    val_dataset = OIST_Loader_Train30(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="val", split=cfg.OIST.split, length=16, delay=1, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=cfg.parameter.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collate_delete_PAD,
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
    
    # Determinism
    torch.backends.cudnn.deterministic = True
    return sum_loss / (batch_idx + 1)

############## validation function ##############
def val(model, val_loader, criterion, device, cfg):
    model.eval()
    coord_F = None
    # Determinism
    torch.backends.cudnn.deterministic = True

    # init setting
    sum_loss = 0
    
    detection_calc_all = Object_Detection(right_range=10, noise_strength=cfg.parameter.noise_strength)


    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            #EP map: targets.size() => [batch,length,H,W]
            #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
            
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            output = model(inputs)
            coord = detection_calc_all.coordinater_maximul(output)
            loss = criterion(output, targets)
            sum_loss += loss.item()
            
            detection_calc_all.calculate(coord, point.flatten(0, 1)[:,:,2:])
    
    acc = detection_calc_all()    
    print("all object accuracy")
    print(acc)


    # Nondeterminism
    torch.backends.cudnn.deterministic = False
    return sum_loss / batch_idx, acc

@hydra.main(config_path="config", config_name='main.yaml')
def main(cfg):
    print("Epoch".ljust(20) + f":{cfg.parameter.epoch}")
    print("Batch size".ljust(20) + f":{cfg.parameter.batch_size}")
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    print("model".ljust(20) + f":{cfg.parameter.model}")
    print("loss".ljust(20) + f":{cfg.parameter.loss}")
    print("alpha".ljust(20) + f":{cfg.parameter.alpha}")
    
    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH_1 = "result/train.txt"
    PATH_2 = "result/validation.txt"

    with open(PATH_1, mode='w') as f:
        pass
    with open(PATH_2, mode='w') as f:
        f.write("epoch\tAccuracy\tPrecision\tRecall\tF1 Score\n")
    
    # data load
    train_loader, val_loader = dataload(cfg)

    move_limit = get_movelimit(train_loader.dataset, factor=1.1)

    # def model
    if cfg.parameter.model == "U-net":
        model = UNet_3D(in_channels=1,
                        n_classes=1,
                        noise_strength=cfg.parameter.noise_strength,
                        delay_upsample=int(math.log2(cfg.parameter.delay)),
                        padding_mode=cfg.parameter.padding).cuda(device)
    elif cfg.parameter.model == "Resnet50":
        model = cd.models.ResNeXt50UNet(in_channels=1, out_channels=1, final_activation=nn.Sigmoid(), pretrained=False, nd=3).cuda(device)

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # def tracking algorithm, accuracy function
    # Track_transfor = MPM_to_Track_normal(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)
    # Track_transfor = MPM_to_Track_detection_copy(noise_strength=cfg.parameter.noise_strength, move_speed=[9., 15.])
    # Track_transfor = MPM_to_Track_multi(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)
    # tracking_fnc = Object_Tracking()

    if cfg.parameter.loss == "MSE":
        criterion = nn.MSELoss()

    elif cfg.parameter.loss == "weighted_MSE":
        criterion = weighted_MSELoss(alpha=None)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.scheduler.max)  # Adam

    scheduler = CosineAnnealingWarmupRestarts(optimizer,
            first_cycle_steps=cfg.scheduler.first,
            cycle_mult=cfg.scheduler.mult,
            max_lr=cfg.scheduler.max,
            min_lr=cfg.scheduler.min,
            warmup_steps=cfg.scheduler.warmup,
            gamma=cfg.scheduler.gamma)
    
    torch.backends.cudnn.benchmark = True

    best_F1 = 0.0
    best_loss = 99999.
    for epoch in range(cfg.parameter.epoch):
        scheduler.step()

        # train
        train_loss = train(model, train_loader, criterion, optimizer, device)
        # validation
        val_loss, val_acc = val(model, val_loader, criterion, device, cfg)
        Accuracy = val_acc["Accuracy"]
        Precition = val_acc["Precition"]
        Recall = val_acc["Recall"]
        F1_Score = val_acc["F1_Score"]

        result_text = f"Epoch{epoch + 1:3d}/{cfg.parameter.epoch:3d} TrainLoss={train_loss:.5f} ValLoss={val_loss:.5f} \
                        Val acc:{Accuracy:.2%}"
        print(result_text)

        ##### Writing result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write(f"{epoch+1}\t{Accuracy:.4f}\t{Precition:.4f}\t{Recall:.4f}\t{F1_Score}\n")
        

        if F1_Score > best_F1:
            best_F1 = F1_Score
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)
        
        ##### result #####
        """
        result_text = "Epoch{:3d}/{:3d} TrainLoss={:.5f} Precision={:.5f} Recall={:.5f} F1 Score={:.5f} ".format(
            epoch + 1,
            cfg.parameter.epoch,
            train_loss,
            Tracking_acc['idp'], Tracking_acc['idr'], Tracking_acc['idf1'])

        print(result_text)

        ##### Writing result #####
        with open(PATH_1, mode='a') as f:
            f.write("{}\t{:.5f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode='a') as f:
            f.write("{epoch+1}\t{1[idp]:.4f}\t{1[idr]:.4f}\t{1[idf1]:.4f}\t{1[num_switches]}\n".format(epoch + 1, Tracking_acc))
        
        if Tracking_acc['idf1'] > best_F1:
            best_F1 = Tracking_acc['idf1']
            PATH = "result/model.pth"
            torch.save(model.state_dict(), PATH)
        """

    print("Best F1Score:{:.4f}%".format(best_F1))

############## main ##############
if __name__ == '__main__':
    main()
