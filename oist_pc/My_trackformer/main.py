#coding: utf-8
#----- 標準ライブラリ -----#
import os
import random
import sys
import time
sys.path.append('../')
#----- 専用ライブラリ -----#
import sacred
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import yaml

#----- 自作ライブラリ -----#
import util.misc as utils
from models import build_model
from datasets import build_dataset
from dataloader import costom_loader
from models.tracker import Tracker
from util.misc import nested_dict_to_namespace
from get_acc import evaluate_mot_accums, get_mot_accum, get_mot_accum_distance


ex = sacred.Experiment('train')
ex.add_config('cfgs/train.yaml')
ex.add_named_config('deformable', 'cfgs/train_deformable.yaml')
ex.add_named_config('tracking', 'cfgs/train_tracking.yaml')
ex.add_named_config('mot17', 'cfgs/train_mot17.yaml')
ex.add_named_config('full_res', 'cfgs/train_full_res.yaml')

tracker_cfg = {
    # [False, 'center_distance', 'min_iou_0_5']
    "public_detections": False,
    # score threshold for detections
    "detection_obj_score_thresh": 0.7,
    # score threshold for keeping the track alive
    "track_obj_score_thresh": 0.8,
    # NMS threshold for detection
    "detection_nms_thresh": 0.9,
    # NMS theshold while tracking
    "track_nms_thresh": 0.9,
    # motion model settings
    # How many timesteps inactive tracks are kept and cosidered for reid
    "inactive_patience": -1,
    # How similar do image and old track need to be to be considered the same person
    "reid_sim_threshold": 0.0,
    "reid_sim_only": False,
    "reid_score_thresh": 0.8,
    "reid_greedy_matching": False
    }


def main(args):
    device = torch.device(args.device)
    split = args.resume.split("/")[1]
    split = "VESICLE_0"
    #RECEPTOR_0
    #VESICLE_0
    # model, criterion, postprocessors
    model, _, postprocessors = build_model(args)
    
    load_model(model, args)

    obj_detect_checkpoint_file = f"save_model/VESICLE_0"
    obj_detect_config_path = os.path.join(
        obj_detect_checkpoint_file,
        'config.yaml')
    obj_detect_args = nested_dict_to_namespace(
        yaml.unsafe_load(open(obj_detect_config_path)))
    img_transform = obj_detect_args.img_transform

    data_loader_val = DataLoader(
        torch.utils.data.Subset(costom_loader(split, img_transform), range(100))
        )
    

    model.to(device)
    model.tracking()

    track_logger = None
    tracker = Tracker(
        model, postprocessors, tracker_cfg,
        False, track_logger)

    tracker.reset()

    #image size 1080x1080
    start = time.time()
    for frame_data in data_loader_val:
        with torch.no_grad():
            tracker.step(frame_data)
    
    end = time.time()
    print("time",end-start)
    

    results = tracker.get_results()

    #track_id,frame,x1,y2,x2,y2
    track = [[track_id, frame, *value['bbox']] for track_id, result_v in results.items() for frame, value in result_v.items() ]
    track = np.array(track)
    np.save("track", track)
    
    print(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
    #mot_accum = get_mot_accum(results, data_loader_val)
    mot_accum = get_mot_accum_distance(results, data_loader_val)
    acc = evaluate_mot_accums(mot_accum)
    
    PATH = f"result/Low_{split}.txt"
    
    with open(PATH, mode='w') as f:
        for k in acc.keys():
            f.write(f"{k}\t")
        f.write("\n")
        for v in acc.values():
            f.write(f"{v}\t")
    
    print(acc)




def load_model(model, args):
    model_without_ddp = model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_state_dict = model_without_ddp.state_dict()
        checkpoint_state_dict = checkpoint['model']
        checkpoint_state_dict = {
            k.replace('detr.', ''): v for k, v in checkpoint['model'].items()}

        resume_state_dict = {}
        for k, v in model_state_dict.items():
            if k not in checkpoint_state_dict:
                resume_value = v
                print(f'Load {k} {tuple(v.shape)} from scratch.')
            elif v.shape != checkpoint_state_dict[k].shape:
                checkpoint_value = checkpoint_state_dict[k]
                num_dims = len(checkpoint_value.shape)

                if 'norm' in k:
                    resume_value = checkpoint_value.repeat(2)
                elif 'multihead_attn' in k or 'self_attn' in k:
                    resume_value = checkpoint_value.repeat(num_dims * (2, ))
                elif 'linear1' in k or 'query_embed' in k:
                    if checkpoint_value.shape[1] * 2 == v.shape[1]:
                        # from hidden size 256 to 512
                        resume_value = checkpoint_value.repeat(1, 2)
                    elif checkpoint_value.shape[0] * 5 == v.shape[0]:
                        # from 100 to 500 object queries
                        resume_value = checkpoint_value.repeat(5, 1)
                    elif checkpoint_value.shape[0] > v.shape[0]:
                        resume_value = checkpoint_value[:v.shape[0]]
                    elif checkpoint_value.shape[0] < v.shape[0]:
                        resume_value = v
                    else:
                        raise NotImplementedError
                elif 'linear2' in k or 'input_proj' in k:
                    resume_value = checkpoint_value.repeat((2,) + (num_dims - 1) * (1, ))
                elif 'class_embed' in k:
                    # person and no-object class
                    # resume_value = checkpoint_value[[1, -1]]
                    # resume_value = checkpoint_value[[0, -1]]
                    # resume_value = checkpoint_value[[1,]]
                    resume_value = v
                else:
                    raise NotImplementedError(f"No rule for {k} with shape {v.shape}.")

                print(f"Load {k} {tuple(v.shape)} from resume model "
                      f"{tuple(checkpoint_value.shape)}.")
            elif args.resume_shift_neuron and 'class_embed' in k:
                checkpoint_value = checkpoint_state_dict[k]
                # no-object class
                resume_value = checkpoint_value.clone()
                # no-object class
                # resume_value[:-2] = checkpoint_value[1:-1].clone()
                resume_value[:-1] = checkpoint_value[1:].clone()
                resume_value[-2] = checkpoint_value[0].clone()
                print(f"Load {k} {tuple(v.shape)} from resume model and "
                      "shift class embed neurons to start with label=0 at neuron=0.")
            else:
                resume_value = checkpoint_state_dict[k]

            resume_state_dict[k] = resume_value

        if args.masks and args.load_mask_head_from_model is not None:
            checkpoint_mask_head = torch.load(
                args.load_mask_head_from_model, map_location='cpu')

            for k, v in resume_state_dict.items():

                if (('bbox_attention' in k or 'mask_head' in k)
                    and v.shape == checkpoint_mask_head['model'][k].shape):
                    print(f'Load {k} {tuple(v.shape)} from mask head model.')
                    resume_state_dict[k] = checkpoint_mask_head['model'][k]

        model_without_ddp.load_state_dict(resume_state_dict)
        print("モデルロード完了")

@ex.main
def load_config(_config, _run):
    """ We use sacred only for config loading from YAML files. """
    sacred.commands.print_config(_run)


if __name__ == '__main__':
    # TODO: hierachical Namespacing for nested dict
    config = ex.run_commandline().config

    args = nested_dict_to_namespace(config)
    main(args)
