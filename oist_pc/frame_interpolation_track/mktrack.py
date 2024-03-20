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
from utils.dataset import OIST_Loader_Exp
import utils.transforms as tf
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_detection_copy, MPM_to_Track_multi
from utils.evaluation import Object_Tracking
from models.Unet_3D import UNet_3D

############## dataloader function ##############
def dataload(cfg):
    test_dataset = OIST_Loader_Exp(root_dir=cfg.OIST.root_dir, Staining=cfg.OIST.Staining, length=8)
    #move_speed_max = OIST_Loader_Test30FPS(Staining="GFP-Mid", mode="train", split=0, length=16).get_max_move_speed()
    #move_speed_max = [4.756889343261719, 5.734686374664307]
    move_speed_max = [9., 15.]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_speed_max


############## test function ##############
def test(cfg, model, test_loader, Track_transfor, device):
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            inputs = inputs.cuda(device, non_blocking=True)
            output = model(inputs)
            # torchvision.utils.save_image(output[0].transpose(0,1),
            #                             "test.png",
            #                             normalize=True)
            # input("save")

            Track_transfor.update(output)
    
    track = Track_transfor(len(test_loader.dataset.get_videos()), delay=2)
    #track = Track_transfor(len(test_loader.dataset.get_videos()), delay=4)
    #assert False,"al"
    video_name = cfg.OIST.Staining.split(".")[0]
    np.savetxt(f"result/{video_name}-track.txt", track, delimiter=',', fmt=["%d", "%d", "%.2f", "%.2f"])



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
    # Track_transfor = MPM_to_Track_normal(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)
    Track_transfor = MPM_to_Track_detection_copy(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)
    # Track_transfor = MPM_to_Track_multi(noise_strength=cfg.parameter.noise_strength, move_speed=move_speed_max)


    Tracking = test(cfg, model, test_loader, Track_transfor, device)



############## main ##############
if __name__ == '__main__':
    main()
