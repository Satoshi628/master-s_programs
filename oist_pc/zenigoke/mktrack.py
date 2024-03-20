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
from models.Correlation_Filter import Wide_ResNet50V2
from utils.dataset import Zenigoke_Loader_Exp, Prompt_Loader
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_detection_copy, MPM_to_Track_multi

############## dataloader function ##############
def dataload(cfg):
    test_dataset = Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name)
    move_speed_max = [3., 6.]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    prompt_dataset = Prompt_Loader(prompt_root_dir=cfg.dataset.prompt_dir)
    prompt_loader = torch.utils.data.DataLoader(
        prompt_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    

    return test_loader, prompt_loader, move_speed_max



############## fit function ##############
def fit(cfg, model, prompt_loader, device):
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(prompt_loader), total=len(prompt_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            inputs = inputs.cuda(device, non_blocking=True)
            inputs = inputs.flatten(0,1)
            output = model.fit(inputs)



############## test function ##############
def test(cfg, model, test_loader, Track_transfor, device):
    model.eval()
    detection = []
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            inputs = inputs.cuda(device, non_blocking=True)
            detect = model(inputs)
            print(len(detect))
            print(len(detect[0]))

            detection.extend(detect)

            Track_transfor.update(detect)
    
    print(len(detection))

    track = Track_transfor(len(test_loader.dataset.get_videos()), delay=1)
    #track = Track_transfor(len(test_loader.dataset.get_videos()), delay=4)
    #assert False,"al"
    video_name = cfg.dataset.video_name.split(".")[0]
    np.savetxt(f"result/{video_name}-track.txt", track, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])



@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    print("detection threshold".ljust(20) + f":{cfg.parameter.threshold}")

    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")


    # data load
    test_loader, prompt_loader, move_speed_max = dataload(cfg)


    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    model = Wide_ResNet50V2(extract_layers=["layer2"], threshold=cfg.parameter.threshold, device=device)
    

    # def tracking algorithm, accuracy function
    # Track_transfor = MPM_to_Track_normal(threshold=cfg.parameter.threshold, move_speed=move_speed_max)
    Track_transfor = MPM_to_Track_detection_copy(threshold=cfg.parameter.threshold, move_speed=move_speed_max)
    # Track_transfor = MPM_to_Track_multi(threshold=cfg.parameter.threshold, move_speed=move_speed_max)


    fit(cfg, model, prompt_loader, device)
    Tracking = test(cfg, model, test_loader, Track_transfor, device)





############## main ##############
if __name__ == '__main__':
    main()
