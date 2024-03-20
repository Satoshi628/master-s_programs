#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
import math

#----- Public Package -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- Module -----#
from utils.dataset import OIST_Loader_Test
import utils.transforms as tf
from utils.MPM_to_track import MPM_to_Track, MPM_to_Track2
from utils.evaluation import Object_Tracking
from models.Unet_3D import UNet_3D

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([tf.Pad(diviser=32)])
    
    test_dataset = OIST_Loader_Test(root_dir="/mnt/kamiya/dataset/NC1032", length=16, transform=test_transform)
    move_speed_max = [20, 20.]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_speed_max


############## test function ##############
def test(model, test_loader, Track_transfor, Tracking_Eva, device):
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            #EP map: targets.size() => [batch,length,H,W]
            #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
            
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            output = model(inputs)

            Track_transfor.update(output)
    
    track = Track_transfor(delay=2)
    #track = Track_transfor(delay=4)
    np.savetxt("result/track.txt", track, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])
    
    Tracking_Eva.update(track, test_loader.dataset.CP_data)
    acc = Tracking_Eva()

    Track_transfor.reset()

    np.savetxt("result/track.txt", track, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])

    return acc


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    print("Batch size".ljust(20) + f":{cfg.parameter.batch_size}")
    print("detection threshold".ljust(20) + f":{cfg.parameter.noise_strength}")
    print("delay".ljust(20) + f":{cfg.parameter.delay}")

    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test.txt"

    with open(PATH, mode='w') as f:
        f.write("")
    
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

    torch.backends.cudnn.benchmark = True

    # def tracking algorithm, accuracy function
    Track_transfor = MPM_to_Track(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)
    #Track_transfor = MPM_to_Track2(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)

    tracking_fnc = Object_Tracking()

    Tracking = test(model, test_loader, Track_transfor, tracking_fnc, device)

    with open(PATH, mode='a') as f:
        for k in Tracking.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in Tracking.values():
            f.write(f"{v}\t")
    
    print("Tracking Evalution")
    for k, v in Tracking.items():
        print(f"{k}".ljust(20) + f":{v}")



############## main ##############
if __name__ == '__main__':
    main()
