#coding: utf-8
#----- Standard Library -----#
import os
import random
import time
import math

#----- Public Package -----#
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import roi_align, nms
import torch.nn as nn
from tqdm import tqdm
#----- Module -----#


class Wide_ResNet50V2_residual(nn.Module):
    def __init__(self, extract_layers=["layer1", "layer2", "layer3"],
            thresh_low=0.3,
            thresh_high=0.6,
            mask_thresh=0.1,
            anchor_sizes=[[12],],
            anchor_aspects=[[1],],
            iou_thresh=0.05,
            device="cuda:0"):
        super().__init__()
        self.extract_layers = extract_layers
        self.device = device
        self.iou_thresh = iou_thresh

        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2).cuda()
        self.extractor = create_feature_extractor(self.model, extract_layers)
        self._set_interpolate()

    def _set_interpolate(self):
        inputs = torch.rand([1, 3, 224, 224], device=self.device)
        features = self.extractor(inputs)
        inter_dict = {}
        for layer_name in self.extract_layers:
            inter_dict[layer_name] = features[layer_name].shape[-2:]

        inter_size = np.array(list(inter_dict.values())).max(axis=0)
        self.inter_rate = inter_size / inputs.shape[-2:]

    def forward(self, x1, x2, x3):
        # batchsize = 1
        if x1 is not None:
            feat1 = self._embed(x1)
        feat2 = self._embed(x2)
        if x3 is not None:
            feat3 = self._embed(x3)
        print(x1.shape)
        print(feat1.shape)
        print(feat2.shape)
        print(self.inter_rate)
        input()



        
        return residual_map, bboxes

    def _embed(self, x):
        input_size = x.shape[-2:]
        feat_size = self.inter_rate * input_size
        feat_size = tuple(feat_size.astype(np.int32))

        features = self.extractor(x)

        feats_list = []
        for layer_name in self.extract_layers:
            feats_list.append(F.interpolate(features[layer_name], size=feat_size))
        
        #[b, C_1+C_2, H_f, W_f]
        features = torch.cat(feats_list, dim=1)
        # features = F.normalize(features)

        return features


def im_read(path, GRAY=False):
    img = cv2.imread(path)
    if GRAY:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return torch.from_numpy(img).float() / 255


def make_seg(image_paths):
    res_list = []
    past_img = None
    pre_img = None

    for path in tqdm(image_paths):
        if past_img is None:
            past_img = im_read(path, GRAY=False).cuda()
            continue
    
        pre_img = im_read(path, GRAY=False).cuda()
        
        # res = torch.abs(pre_img - past_img)
        res = pre_img - past_img
        res_list.append(res.cpu())
        past_img = pre_img
    
    reses = torch.stack(res_list, dim=0)
    print(reses.shape)
    reses = reses.max(dim=0)[0]
    torchvision.utils.save_image(reses[None],
                                "seg.png",
                                normalize=True)


#28,248.05326601862907,752.7761235237122
#48, 905.0690097808838, 576.4227186441422
def correlation_filter(img):
    
    X, Y = 905, img.shape[0] - 576

    kernelsize = 21
    weights = img[Y - kernelsize // 2 - 1 : Y + kernelsize // 2, X - kernelsize // 2 - 1 : X + kernelsize // 2]
    weights = weights[None, None]

    torchvision.utils.save_image(weights,
                                 "weights.png",
                                 normalize=True)
    
    img = F.conv2d(img[None], weights, padding=(kernelsize - 1) // 2) / (kernelsize * kernelsize)

    # img = 1. * (img > 0.2)

    torchvision.utils.save_image(img[None],
                                 "corre.png",
                                 normalize=True)

def feature_residual(img1, img2, img3, tracks):
    model = Wide_ResNet50V2_residual()
    model.cuda()
    model.eval()
    residual_map, bboxes = model(img1[None], img2[None], img3[None], tracks)

    img2 = torchvision.utils.draw_bounding_boxes((img2*255).to(torch.uint8), bboxes[:, :-1], width=2)
    
    torchvision.utils.save_image(residual_map,
                                "residual_map.png",
                                normalize=True)
    torchvision.utils.save_image(img2[None].float(),
                                "detection.png",
                                normalize=True)



image_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/code/my_exp/kakon/video_images/2x_sample1", "*")))
gray_img_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/code/my_exp/kakon/video_images/focusedXY05_8bit", "*")))

#大津の二値か
_, th_img = cv2.threshold(cv2.cvtColor(cv2.imread(gray_img_paths[-1]), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("threshold.png", th_img)

inputs1 = im_read(image_paths[29]).cuda()
inputs2 = im_read(image_paths[30]).cuda()
inputs3 = im_read(image_paths[31]).cuda()

track_paths = sorted(glob.glob(os.path.join("/mnt/kamiya/dataset/pseudoroot/20 tracks", "*")))
H, W = inputs1.shape[-2:]

tracks = []
for idx, track_path in enumerate(track_paths):
    track = np.loadtxt(track_path, delimiter=',').astype(np.int32)
    track[:, 2] = H - track[:, 2]
    track = np.concatenate([track[:, :1], np.ones_like(track[:, :1]) * idx, track[:, -2:]], axis=-1)
    tracks.append(track)

tracks = np.concatenate(tracks, axis=0)
tracks = torch.from_numpy(tracks).cuda()

gray_latest = im_read(gray_img_paths[31], GRAY=True).cuda()
feature_residual(inputs1, inputs2, inputs3, tracks)
res = torch.abs(inputs1 - inputs2)
res = (res > 0.7) * 1.

images = torch.stack([inputs1, inputs2, res], dim=0)

torchvision.utils.save_image(images,
                             "test.png",
                             normalize=True)

# make_seg(image_paths)
# correlation_filter(im_read(image_paths[48+1]).cuda())
