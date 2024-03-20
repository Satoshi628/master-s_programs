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
            anchor_sizes=[[5],],
            anchor_aspects=[[1.0],],
            iou_thresh=0.05,
            device="cuda:0"):
        super().__init__()
        self.extract_layers = extract_layers
        self.device = device
        self.iou_thresh = iou_thresh

        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2).cuda()
        self.extractor = create_feature_extractor(self.model, extract_layers)
        self._set_interpolate()

        anchor_sizes = [self.inter_rate[0]*t for tuples in anchor_sizes for t in tuples]
        self.anchorgen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspects)
        
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high
        self.mask_thresh = mask_thresh

    def _set_interpolate(self):
        inputs = torch.rand([1, 3, 224, 224], device=self.device)
        features = self.extractor(inputs)
        inter_dict = {}
        for layer_name in self.extract_layers:
            inter_dict[layer_name] = features[layer_name].shape[-2:]

        inter_size = np.array(list(inter_dict.values())).max(axis=0)
        self.inter_rate = inter_size / inputs.shape[-2:]

    def forward(self, x1, x2, x3, gray):
        # batchsize = 1
        if x1 is not None:
            feat1 = self._embed(x1)
        feat2 = self._embed(x2)
        if x3 is not None:
            feat3 = self._embed(x3)

        if x1 is not None:
            residual_map12 = (feat1 - feat2).sum(dim=1, keepdim=True).abs()
        if x3 is not None:
            residual_map23 = (feat2 - feat3).sum(dim=1, keepdim=True).abs()
        
        if x1 is not None and x3 is not None:
            residual_map = (residual_map12 + residual_map23)/2
        elif x1 is not None:
            residual_map = residual_map12
        elif x3 is not None:
            residual_map = residual_map23

        residual_map_max = residual_map.flatten(1).max(dim=-1)[0][:, None, None, None]
        residual_map_min = residual_map.flatten(1).min(dim=-1)[0][:, None, None, None]
        residual_map = (residual_map - residual_map_min) / (residual_map_max - residual_map_min + 1e-6)

        # residual_map = F.interpolate(residual_map, size=tuple(x1.shape[-2:]))
        # residual_map = (residual_map > self.thresh_low) * 1.

        images = ImageList(tensors=residual_map, image_sizes=[tuple(r.shape[-2:]) for r in residual_map])
        anchors = self.anchorgen(images, [r for r in residual_map])
        logits = roi_align(residual_map, boxes=anchors, output_size=1)

        if gray is not None:
            mask_logits = roi_align(gray, boxes=[anchors[0] / self.inter_rate[0]], output_size=1)

            mask_logits = mask_logits.flatten(0)
            logits_max = mask_logits.max()
            mask_logits = mask_logits / (logits_max + 1e-6)
        else:
            mask_logits = 1.
        
        logits = logits.flatten(0)
        logits_max = logits.max()
        logits = logits / (logits_max + 1e-6)

        flags = (logits > self.thresh_low) & (logits < self.thresh_high) & (mask_logits > self.mask_thresh)
        flags = (logits > self.thresh_low) & (logits < self.thresh_high)
        anchors_low = anchors[0][flags]
        logits_low = logits[flags]

        indices = nms(anchors_low, logits_low, 0.1)
        anchors_low = anchors_low[indices]
        logits_low = logits_low[indices]

        anchors_high = anchors[0][logits > self.thresh_high]
        logits_high = logits[logits > self.thresh_high]
        
        indices = nms(anchors_high, logits_high, self.iou_thresh)
        anchors_high = anchors_high[indices]
        logits_high = logits_high[indices]



        bboxes = torch.cat([anchors_high, anchors_low], dim=0)
        score = torch.cat([logits_high, logits_low], dim=0)
        indices = nms(bboxes, score, self.iou_thresh)
        bboxes = bboxes[indices]
        score = score[indices]

        #元の画像の大きさの戻す
        bboxes = bboxes / self.inter_rate[0]

        bboxes = torch.cat([bboxes, score[:, None]], dim=-1)
        
        # torchvision.utils.save_image(logits.reshape(1, 128, 128, 5).permute(3, 0, 1, 2),
        #                     "test2.png",
        #                     normalize=True)
        
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


class ResidualMap(nn.Module):
    def __init__(self,
            thresh_low=0.3,
            thresh_high=0.6,
            anchor_sizes=[[5],],
            anchor_aspects=[[1.0],],
            iou_thresh=0.05,
            device="cuda:0"):
        super().__init__()
        self.device = device
        self.iou_thresh = iou_thresh

        anchor_sizes = [t for tuples in anchor_sizes for t in tuples]
        self.anchorgen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspects)
        
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

    def forward(self, x1, x2, x3, gray):
        if x1 is not None:
            residual_map12 = torch.abs(x1 - x2).sum(dim=1)
        if x3 is not None:
            residual_map23 = torch.abs(x2 - x3).sum(dim=1)
        
        if x1 is not None and x3 is not None:
            residual_map = (residual_map12 + residual_map23)/2
        elif x1 is not None:
            residual_map = residual_map12
        elif x3 is not None:
            residual_map = residual_map23

        residual_map_max = residual_map.flatten(1).max(dim=-1)[0][:, None, None, None]
        residual_map_min = residual_map.flatten(1).min(dim=-1)[0][:, None, None, None]
        residual_map = (residual_map - residual_map_min) / (residual_map_max - residual_map_min + 1e-6)


        images = ImageList(tensors=residual_map, image_sizes=[tuple(r.shape[-2:]) for r in residual_map])
        anchors = self.anchorgen(images, [r for r in residual_map])
        logits = roi_align(residual_map, boxes=anchors, output_size=1)

        
        logits = logits.flatten(0)
        logits_max = logits.max()
        logits = logits / (logits_max + 1e-6)

        flags = (logits > self.thresh_low) & (logits < self.thresh_high)
        anchors_low = anchors[0][flags]
        logits_low = logits[flags]

        indices = nms(anchors_low, logits_low, 0.1)
        anchors_low = anchors_low[indices]
        logits_low = logits_low[indices]

        anchors_high = anchors[0][logits > self.thresh_high]
        logits_high = logits[logits > self.thresh_high]
        
        indices = nms(anchors_high, logits_high, self.iou_thresh)
        anchors_high = anchors_high[indices]
        logits_high = logits_high[indices]



        bboxes = torch.cat([anchors_high, anchors_low], dim=0)
        score = torch.cat([logits_high, logits_low], dim=0)
        indices = nms(bboxes, score, self.iou_thresh)
        bboxes = bboxes[indices]
        score = score[indices]

        bboxes = torch.cat([bboxes, score[:, None]], dim=-1)
        
        # torchvision.utils.save_image(logits.reshape(1, 128, 128, 5).permute(3, 0, 1, 2),
        #                     "test2.png",
        #                     normalize=True)
        
        return residual_map, bboxes


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

def feature_residual(img1, img2, img3, gray_latest):
    model = Wide_ResNet50V2_residual()
    # model = ResidualMap()
    model.cuda()
    # model.eval()
    # residual_map, bboxes = model(img1[None], img2[None], None, gray_latest[None,None])
    residual_map, bboxes = model(img1[None], img2[None], img3[None], None)

    img2 = torchvision.utils.draw_bounding_boxes((img2*255).to(torch.uint8), bboxes[:, :-1], width=1)
    
    torchvision.utils.save_image(residual_map,
                                "residual_map.png",
                                normalize=True)
    torchvision.utils.save_image(img2[None].float(),
                                "detection.png",
                                normalize=True)


image_paths = sorted(glob.glob(os.path.join("videos/Hyp-t8-1/images", "*")))

# _, th_img = cv2.threshold(cv2.cvtColor(cv2.imread(gray_img_paths[-1]), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
# cv2.imwrite("threshold.png", th_img)

inputs1 = im_read(image_paths[0]).cuda()
inputs2 = im_read(image_paths[1]).cuda()
inputs3 = im_read(image_paths[2]).cuda()
feature_residual(inputs1, inputs2, inputs3, None)
res = torch.abs(inputs1 - inputs2)
# res = (res > 0.3) * 1.

images = torch.stack([inputs1, inputs2, res], dim=0)

torchvision.utils.save_image(images,
                             "test.png",
                             normalize=True)

# make_seg(image_paths)
# correlation_filter(im_read(image_paths[48+1]).cuda())
