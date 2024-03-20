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
from utils.dataset import OISTLoader, OISTLoader_raw_data, collate_delete_PAD
import utils.transforms as tf
from utils.utils import get_movelimit
from utils.Transformer_to_track import Transformer_to_Track
from utils.evaluation import Object_Detection,Object_Tracking
from models.PTGT import PTGT

############## dataloader function ##############
def dataload(cfg):
    test_dataset = OISTLoader_raw_data(data=cfg.parameter.data, frame=cfg.parameter.frame)
    mode_limit_dataset = OISTLoader(mode="train", data="GFP_immobile", length=16)
    move_limit = get_movelimit(mode_limit_dataset)

    return test_dataset, move_limit

############## test function ##############
def test(model, test_loader, TtT, device):
    model.eval()
    coord_F = None

    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs.cuda(device, non_blocking=True)

            vector, coord, coord_F = model.tracking_process(inputs, add_F_dict=coord_F)

            # update Tracking
            TtT.update(vector, coord)

        TtT()
        TtT.save_track()

@hydra.main(config_path="config", config_name='analysis.yaml')
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

    test(model, test_loader, TtT, device)



############## main ##############
if __name__ == '__main__':
    main()
