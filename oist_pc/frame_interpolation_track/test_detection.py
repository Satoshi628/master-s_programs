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
from utils.dataset import OIST_Loader, OIST_Loader_Train30_2, collate_delete_PAD
import utils.transforms as tf
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_detection_copy, MPM_to_Track_multi
from utils.evaluation import Object_Detection
from models.Unet_3D import UNet_3D
from models.Unet_2D import UNet_2D

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([])
    test_dataset = OIST_Loader_Train30_2(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, mode="test", split=cfg.OIST.split, length=16, delay=cfg.parameter.delay, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader


############## test function ##############
def test(model, test_loader, device, cfg):
    model.eval()

    # init setting
    
    detection_calc = Object_Detection(right_range=10, noise_strength=cfg.parameter.noise_strength)


    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            #EP map: targets.size() => [batch,length,H,W]
            #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
            
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)

            output, coord = model.get_coord(inputs)
            
            detection_calc.calculate(coord, point.flatten(0, 1)[:, :, -2:])
    

    acc = detection_calc()    
    print("all object accuracy")
    print(acc)
    
    return acc


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
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
    test_loader = dataload(cfg)

    # def model
    if cfg.parameter.use_3D:
        model = UNet_3D(in_channels=1,
                        n_classes=1,
                        noise_strength=cfg.parameter.noise_strength,
                        delay_upsample=int(math.log2(cfg.parameter.delay)),
                        padding_mode=cfg.parameter.padding).cuda(device)
    else:
        model = UNet_2D(in_channels=1,
                    n_classes=1,
                    noise_strength=cfg.parameter.noise_strength,
                    delay_upsample=int(math.log2(cfg.parameter.delay)),
                    padding_mode=cfg.parameter.padding).cuda(device)
    
    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    torch.backends.cudnn.benchmark = True

    Tracking = test(model, test_loader, device, cfg)

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
