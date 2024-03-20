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
from utils.dataset import Zenigoke_Loader, collate_delete_PAD
import utils.transforms as tf
from utils.evaluation import Object_Detection, Object_Tracking, Calc_AP
from models.Unet_2D import UNet_2D
from models.Unet_3D import UNet_3D
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_kalman, MPM_to_Track_multi, global_association_solver

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([])
    test_dataset = Zenigoke_Loader(root_dir=cfg.dataset.root_dir, mode="test", split=cfg.dataset.split, length=16, delay=cfg.parameter.delay, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
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
    detection = []
    tracks = []
    
    detection_calc = Object_Detection(right_range=10)
    calc_ap = Calc_AP(right_range=[10., 15.])
    tracking_calc = Object_Tracking(right_range=10)
    tracker = MPM_to_Track_kalman(move_speed=[11.77, 11.77], init_move=[0.0, 0.0])

    # tracker.update([torch.empty([0, 3])])

    for batch_idx, (inputs, targets, point) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
        #image: input.size() => [batch,channel,length,H,W]
        #EP map: targets.size() => [batch,length,H,W]
        #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
        
        inputs = inputs.cuda(device, non_blocking=True)
        # targets = targets.cuda(device, non_blocking=True)

        output, coord = model.get_coord(inputs)
        detection.extend(coord)
        # print(output.max())
        # print(output.min())
        # torchvision.utils.save_image(output[0].transpose(0,1), "images.png", normalize=True)
        # input()
        detection_calc.update([c[:, [0, 1]] for c in coord], point[0])
        calc_ap.update(output, point[0])

        tracker.update(coord)


    detection = [torch.cat([torch.full([dets.shape[0], 1], f), dets], dim=-1) for f, dets in enumerate(detection)]
    detection = torch.cat(detection, dim=0)
    np.savetxt(f"result/dets.txt", detection.numpy(), delimiter=',', fmt=["%d", "%d", "%d", "%.2f"])

    acc = detection_calc()
    print("detection accuracy")
    print(acc)
    
    acc = calc_ap()
    print("all object accuracy")
    print(acc)
    tracks = tracker(len(test_loader.dataset.get_videos()), delay=1) #.cpu().numpy()
    tracks = global_association_solver(tracks)
    
    label_track = test_loader.dataset.CP_batch_data

    
    tracking_calc.update(tracks, label_track, len(test_loader.dataset.images))
    acc = tracking_calc()
    
    video_name = os.path.splitext(os.path.basename(test_loader.dataset.video_file))[0]
    tracks = np.concatenate([tracks[:, :2], tracks[:, 2:4] - 6, tracks[:, 2:4] + 6, tracks[:, -1:]], axis=-1)
    np.savetxt(f"result/track.txt", tracks, delimiter=',', fmt=["%d", "%d", "%d", "%d", "%d", "%d", "%.2f"])
    # np.savetxt(f"result/{video_name}-track.txt", tracks, delimiter=',', fmt=["%d", "%d", "%d", "%d", "%d", "%d", "%.2f"])
    return acc


@hydra.main(config_path="config", config_name='test_detection.yaml')
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
        model = UNet_3D(in_channels=3,
                        n_classes=1,
                        channel=32,
                        noise_strength=cfg.parameter.noise_strength,
                        delay_upsample=int(math.log2(cfg.parameter.delay)),
                        padding_mode=cfg.parameter.padding).cuda(device)
    else:
        model = UNet_2D(in_channels=3,
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
