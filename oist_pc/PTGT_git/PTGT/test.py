#coding: utf-8
#----- Standard Library -----#
import os
import random
import time

#----- Public Package -----#
import hydra
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

#----- Module -----#
from utils.dataset import OISTLoader, OISTLoader_add, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.Transformer_to_track import Transformer_to_Track
from utils.evaluation import Object_Detection,Object_Tracking
from models.PTGT import PTGT

############## dataloader function ##############
def dataload(cfg):
    ### data augmentation + preprocceing ###
    test_transform = tf.Compose([])
    
    test_dataset = OISTLoader_add(mode="test", split=0, length=16, transform=test_transform)
    mode_limit_dataset = OISTLoader(mode="train", split=0, length=16)
    move_limit = get_movelimit(mode_limit_dataset)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_delete_PAD,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    
    return test_loader, move_limit

############## test function ##############
def test(model, test_loader, TtT, device):
    model.eval()
    coord_F = None

    # def accuracy function
    tracking_fnc = Object_Tracking()
    detection_fnc = Object_Detection()

    with torch.no_grad():
        for batch_idx, (inputs, targets, point) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            #EP map: targets.size() => [batch,length,H,W]
            #object coordinate: point.size() => [batch,length,num,4(frame,id,x,y)]
            inputs = inputs.cuda(device, non_blocking=True)
            point = point.cuda(device, non_blocking=True)
            point = point.long()

            vector, coord, coord_F = model.tracking_process(inputs, add_F_dict=coord_F)

            # update Tracking
            TtT.update(vector, coord)
            # update Detection
            detection_fnc.calculate(coord.flatten(0, 1), point[:, :, :, 2:].flatten(0, 1))

        Detection_acc = detection_fnc()

        TtT()
        TtT.save_track()
        tracking_fnc.update(TtT.get_track(), test_loader.dataset.CP_data)
        Tracking_acc = tracking_fnc()

    return Tracking_acc, Detection_acc

@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    print("Assignment".ljust(20) + f":{cfg.parameter.assignment}")
    print("Number of feature".ljust(20) + f":{cfg.parameter.feature_num}")
    print("Overlap range".ljust(20) + f":{cfg.parameter.overlap_range}")
    print("Positonal encoding".ljust(20) + f":{cfg.parameter.pos}")
    print("Attention".ljust(20) + f":{cfg.parameter.encoder}")
    print("Detection threshold".ljust(20) + f":{cfg.parameter.noise_strength}")
    print("Backbone path".ljust(20) + f":{cfg.parameter.back_bone}")



    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")

    PATH = "result/test.txt"

    with open(PATH, mode='w') as f:
        f.write("")
    
    # data load
    test_loader, move_limit = dataload(cfg)

    # def model
    model = PTGT(back_bone_path=cfg.parameter.back_bone,
                    noise_strength=cfg.parameter.noise_strength,
                    pos_mode=cfg.parameter.pos,
                    assignment=cfg.parameter.assignment,
                    feature_num=cfg.parameter.feature_num,
                    overlap_range=cfg.parameter.overlap_range,
                    encode_mode=cfg.parameter.encoder,
                    move_limit=move_limit[0]).cuda(device)

    model_path = "result/model.pth"
    model.load_state_dict(torch.load(model_path))

    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch

    torch.backends.cudnn.benchmark = True

    TtT = Transformer_to_Track(tracking_mode="P", move_limit=move_limit)

    Tracking, Detection = test(model, test_loader, TtT, device)

    with open(PATH, mode='a') as f:
        for k in Detection.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in Detection.values():
            f.write(f"{v}\t")
        f.write("\n")
        for k in Tracking.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in Tracking.values():
            f.write(f"{v}\t")
    
    #Accuracy, Precition, Recall, F1_Score
    print("Detection Evalution")
    for k, v in Detection.items():
        print(f"{k}".ljust(20) + f":{v}")

    print("Tracking Evalution")
    for k, v in Tracking.items():
        print(f"{k}".ljust(20) + f":{v}")



############## main ##############
if __name__ == '__main__':
    main()
