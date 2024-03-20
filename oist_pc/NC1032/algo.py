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
from utils.loss import connected_loss
from utils.dataset import OIST_Loader, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.MPM_to_track import MPM_to_Track, MPM_to_Track2
from utils.evaluation import Object_Tracking
from models.Unet_3D import UNet_3D



############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([])

    test_dataset = OIST_Loader(Staining="GFP-Mid", mode="test", split=0, length=16, delay=cfg.parameter.delay, transform=test_transform)
    #move_speed_max = OIST_Loader(Staining="GFP-Mid", mode="train", split=0, length=16).get_max_move_speed()
    move_speed_max = [4.756889343261719, 5.734686374664307]

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_speed_max

############## validation function ##############
def test(model, test_loader, Track_transfor, Tracking_Eva, device):
    model.eval()
    coord_F = None
    # Determinism
    torch.backends.cudnn.deterministic = True

    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            #EP map: targets.size() => [batch,length,H,W]
            #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
            print(inputs.shape)
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            output = model(inputs)

            Track_transfor.update(output)
    
    track = Track_transfor()
    #assert False,"al"
    Tracking_Eva.update(track, test_loader.dataset.CP_data)
    acc = Tracking_Eva()

    Track_transfor.reset()
    print(acc)

    # Nondeterminism
    torch.backends.cudnn.deterministic = False
    return acc

@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("Batch size".ljust(20) + f":{cfg.parameter.batch_size}")
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    
    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")
    
    # data load
    test_loader, move_speed_max = dataload(cfg)


    # def model
    model = UNet_3D(in_channels=1,
                    n_classes=1,
                    noise_strength=cfg.parameter.noise_strength,
                    delay_upsample=int(math.log2(cfg.parameter.delay))).cuda(device)
    
    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    # def tracking algorithm, accuracy function
    Track_transfor = MPM_to_Track(noise_strength=0.4, move_speed=move_speed_max)
    #Track_transfor = MPM_to_Track2(noise_strength=0.4, move_speed=move_speed_max)

    tracking_fnc = Object_Tracking()


    torch.backends.cudnn.benchmark = True

    test(model, test_loader, Track_transfor, tracking_fnc, device)


############## main ##############
if __name__ == '__main__':
    main()
