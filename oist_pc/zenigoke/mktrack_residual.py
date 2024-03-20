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
from models.Correlation_Filter import Wide_ResNet50V2_residual
from tracker.byte_tracker import BYTETracker
from utils.dataset import Zenigoke_Loader_Exp
from utils.MPM_to_track import MPM_to_Track_normal, MPM_to_Track_detection_copy, MPM_to_Track_multi

############## dataloader function ##############
def dataload(cfg):
    test_dataset = Zenigoke_Loader_Exp(root_dir=cfg.dataset.root_dir, video_name=cfg.dataset.video_name)
    move_speed_max = [8., 16.]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False)

    return test_loader, move_speed_max

def STrack2numpy(stracks, time_idx):
    tracks = []
    for t in stracks:
        #x1,y1,x2,y2
        tlbr = t.tlbr
        tid = t.track_id
        score = t.score
        tracks.append(np.array([time_idx, tid, *list(tlbr), score]))
    
    if len(tracks) == 0:
        return np.empty([0, 7])
    tracks = np.stack(tracks)
    return tracks

############## test function ##############
def test(cfg, model, test_loader, tracker, device):
    #なぜかevalモードだとうまくいかない
    # model.eval()
    detection = []
    tracks = []

    x_past = None
    x_pre = None
    x_next = None
    #処理が一つずれるの注意
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(test_loader), total=len(test_loader), leave=False):
            #image: input.size() => [batch,channel,length,H,W]
            img_size = inputs.shape[-2:]

            inputs = inputs.cuda(device, non_blocking=True)
            x_next = inputs
            if x_pre is None:
                x_pre = x_next
                x_past = x_pre
                continue
            detect = model(x_past, x_pre, x_next)
            save_det = detect.detach().cpu().numpy()
            times = np.full([save_det.shape[0], 1], batch_idx-1, dtype=np.int32)
            detection.append(np.concatenate([times, save_det], axis=-1))
            
            detect = detect.detach().cpu()

            STracks = tracker.update(detect, img_size, img_size)
            tracks.append(STrack2numpy(STracks, batch_idx - 1))

            x_pre = x_next
            x_past = x_pre
    
        detect = model(x_past, x_pre, None)
        save_det = detect.detach().cpu().numpy()
        times = np.full([save_det.shape[0], 1], batch_idx, dtype=np.int32)
        detection.append(np.concatenate([times, save_det], axis=-1))

        detect = detect.detach().cpu()
        STracks = tracker.update(detect, img_size, img_size)
        tracks.append(STrack2numpy(STracks, batch_idx))

    detection = np.concatenate(detection, axis=0)
    tracks = np.concatenate(tracks, axis=0)
    
    #assert False,"al"
    video_name = cfg.dataset.video_name.split(".")[0]
    np.savetxt(f"result/{video_name}-dets.txt", detection, delimiter=',', fmt=["%d", "%d", "%d", "%d", "%d", "%.2f"])
    np.savetxt(f"result/{video_name}-track.txt", tracks, delimiter=',', fmt=["%d", "%d", "%d", "%d", "%d", "%d", "%.2f"])


@hydra.main(config_path="config", config_name='test.yaml')
def main(cfg):
    print("GPU".ljust(20) + f":{cfg.parameter.gpu}")
    print("detection bbox anchor size".ljust(30) + f":{cfg.parameter.bbox_anchor_size}")
    print("detection bbox anchor aspect".ljust(30) + f":{cfg.parameter.bbox_anchor_aspects}")
    print("detection threshold low".ljust(30) + f":{cfg.parameter.thresh_low}")
    print("detection threshold high".ljust(30) + f":{cfg.parameter.thresh_high}")
    print("ByteTrack track thresh".ljust(30) + f":{cfg.bytetrack.track_thresh}")
    print("ByteTrack match thresh".ljust(30) + f":{cfg.bytetrack.match_thresh}")

    ##### GPU setting #####
    device = torch.device('cuda:{}'.format(cfg.parameter.gpu) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("result"):
        os.mkdir("result")


    # data load
    test_loader, move_speed_max = dataload(cfg)


    # random number fixing
    rand_seed = 0
    random.seed(rand_seed)  # python
    np.random.seed(rand_seed)  # numpy
    torch.manual_seed(rand_seed)  # pytorch
    
    model = Wide_ResNet50V2_residual(extract_layers=list(cfg.parameter.feature_extract_layers),
                                    thresh_low=cfg.parameter.thresh_low,
                                    thresh_high=cfg.parameter.thresh_high,
                                    anchor_sizes=[list(cfg.parameter.bbox_anchor_size),],
                                    anchor_aspects=[list(cfg.parameter.bbox_anchor_aspects),],
                                    iou_thresh=cfg.parameter.mns_iou_thresh,
                                    device=device)
    

    # def tracking algorithm, accuracy function
    
    tracker = BYTETracker(cfg.bytetrack)
    Tracking = test(cfg, model, test_loader, tracker, device)





############## main ##############
if __name__ == '__main__':
    main()
