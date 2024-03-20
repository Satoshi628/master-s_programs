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
from tqdm import tqdm
import cv2

#----- Module -----#
from utils.dataset import Konmo_Loader, Konmo_Video_Loader
import utils.transforms_seg as tf
from models.Unet_2D import UNet
from utils.MPM_to_track import MPM_to_Track_kalman, global_association_solver, TrackInterpolate

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([
                                tf.Padding_div(32)])

    test_dataset = Konmo_Video_Loader(root_dir=cfg.dataset.root_dir, crop=[1500, 0, 2198, 4592], data_num=None, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    
    return test_loader


############## test function ##############
@torch.no_grad()
def visual(model, test_loader, device, cfg):
    model.eval()
    
    if not os.path.exists("video"):
        os.mkdir("video")
    
    for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
        #元画像の大きさは(分割数+1)*split//2
        _,S,_,H,W = inputs.shape
        ori_img_size = (S+1)*H//2
        ori_image = torch.zeros([3, ori_img_size,W])
        ori_output = torch.zeros([2, ori_img_size,W])

        #[b(1), split, 3, H, W]
        for idx, image in enumerate(inputs[0]):
            ori_image[:, H//2*idx:H//2*(idx+1) + 256] = image
            image = image.cuda(device, non_blocking=True)
            output = model(image[None])[0]
            ori_output[:, H//2*idx:H//2*(idx+1) + 256] += output.cpu()
        
        mask = ori_output.max(dim=0)[1]
        mask = torch.stack([mask, mask, mask], dim=0)
        heatmap = F.softmax(ori_output,dim=0)[1]
        heatmap = torch.stack([heatmap, heatmap, heatmap], dim=0)
        images = torch.stack([ori_image, mask, heatmap], dim=0)

        torchvision.utils.save_image(images,
                                    f"video/{batch_idx:03}.png",
                                    normalize=True)
        if batch_idx == 3:
            break


############## test function ##############
@torch.no_grad()
def track(model, test_loader, device, cfg):
    model.eval()

    Tracker = MPM_to_Track_kalman(move_speed=[15., 15.], init_move=[0., 0.])
    
    if not os.path.exists("video"):
        os.mkdir("video")

    def vis_corner(mask, corners):
        mask = np.stack([mask,mask,mask],axis=-1)
        for corner in corners:
            x,y = corner
            cv2.drawMarker(mask, [int(x),int(y)], (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=4, thickness=1, line_type=cv2.LINE_8)
        # mask = cv2.resize(mask,None , fx=0.5, fy=0.5)
        cv2.imwrite("corner.png", mask)
        input("owa")
    

    for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
        #元画像の大きさは(分割数+1)*split//2
        _,S,_,H,W = inputs.shape
        ori_img_size = (S+1)*H//2
        ori_output = torch.zeros([2, ori_img_size,W])


        #[b(1), split, 3, H, W]
        for idx, image in enumerate(inputs[0]):
            image = image.cuda(device, non_blocking=True)
            output = model(image[None])[0]
            ori_output[:, H//2*idx:H//2*(idx+1) + H//2] += output.cpu()
        
        mask = ori_output.max(dim=0)[1].numpy()
        mask = (mask*255).astype(np.uint8)
        corners = cv2.goodFeaturesToTrack(mask,0,0.3,5)
        corners = corners[:,0]
        Tracker.update([torch.from_numpy(corners)])
    
    tracks = Tracker(len(test_loader), delay=1)#.cpu().numpy()
    tracks = global_association_solver(tracks)
    tracks = TrackInterpolate(tracks)

    np.savetxt(f"result/track.txt", tracks, delimiter=',', fmt=["%d", "%d", "%d", "%d"])

@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")

    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')
    
    # data load
    test_loader = dataload(cfg)

    # def model
    model = UNet(in_channels=3,
                n_classes=2).cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    torch.backends.cudnn.benchmark = True

    # visual(model, test_loader, device, cfg)
    track(model, test_loader, device, cfg)

############## main ##############
if __name__ == '__main__':
    main()
