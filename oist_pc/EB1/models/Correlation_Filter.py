#coding: utf-8
#----- Standard Library -----#
#None

#----- Public Package -----#
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Normalize
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import roi_align, nms
#----- Module -----#
#None


class Wide_ResNet50V2_residual(nn.Module):
    def __init__(self, extract_layers=["layer1", "layer2", "layer3"],
            thresh_low=0.3,
            thresh_high=0.6,
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

        anchor_sizes = [self.inter_rate[0]*t for tuples in anchor_sizes for t in tuples]
        self.anchorgen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspects)
        
        self.thresh_low = thresh_low
        self.thresh_high = thresh_high

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
        logits = logits.flatten(0)
        logits_max = logits.max()
        logits_min = logits.min()
        logits = (logits - logits_min) / (logits_max - logits_min + 1e-6)
        
        anchors_low = anchors[0][(logits > self.thresh_low) & (logits < self.thresh_high)]
        logits_low = logits[(logits > self.thresh_low) & (logits < self.thresh_high)]

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
        
        return bboxes

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

    def get_coord(self, x1, x2, x3):
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
        logits = logits.flatten(0)
        logits_max = logits.max()
        logits_min = logits.min()
        logits = (logits - logits_min) / (logits_max - logits_min + 1e-6)
        
        anchors_low = anchors[0][(logits > self.thresh_low) & (logits < self.thresh_high)]
        logits_low = logits[(logits > self.thresh_low) & (logits < self.thresh_high)]

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

        #x_min, y_min, x_max, y_max => center_x, center_y
        points = (bboxes[:,[0,1]] + bboxes[:,[2,3]])/2

        points = torch.cat([points, score[:, None]], dim=-1)
        
        # torchvision.utils.save_image(logits.reshape(1, 128, 128, 5).permute(3, 0, 1, 2),
        #                     "test2.png",
        #                     normalize=True)
        
        return residual_map, points

class Wide_ResNet50V2(nn.Module):
    def __init__(self, extract_layers=["layer2", "layer3"], noise_strength=0.1, device="cuda:0"):
        super().__init__()
        self.extract_layers = extract_layers
        self.device = device

        self.maxpool = nn.MaxPool2d(5, stride=1, padding=(5 - 1) // 2)
        self.gaussian_filter = torchvision.transforms.GaussianBlur((13, 13), sigma=3)
        self.noise_strength = noise_strength

        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2).cuda()
        self.extractor = create_feature_extractor(self.model, extract_layers)
        self._set_interpolate()
        self.prompts = []

    def _set_interpolate(self):
        inputs = torch.rand([1, 3, 224, 224], device=self.device)
        features = self.extractor(inputs)
        inter_dict = {}
        for layer_name in self.extract_layers:
            inter_dict[layer_name] = features[layer_name].shape[-2:]

        inter_size = np.array(list(inter_dict.values())).max(axis=0)
        self.inter_rate = inter_size / inputs.shape[-2:]

    def reset(self):
        self.prompts = []

    def forward(self, x):
        #[N,C,H_f,W_f]
        prompts = torch.cat(self.prompts, dim=0).cuda(self.device)

        input_size = x.shape[-2:]
        feat_size = self.inter_rate * input_size
        feat_size = tuple(feat_size.astype(np.int32))

        features = self.extractor(x)

        feats_list = []
        for layer_name in self.extract_layers:
            feats_list.append(F.interpolate(features[layer_name], size=feat_size))
        
        #[b, C_1+C_2, H_f, W_f]
        features = torch.cat(feats_list, dim=1)
        features = F.normalize(features)

        #feature => [b,C,H_f1,W_f1]
        #prompts => [N,C,H_f,W_f]
        pad = [f // 2 for f in prompts.shape[-2:]]
        kernel_area = prompts.shape[-1] * prompts.shape[-2]
        # correlation => [b,1,H,W]
        correlation = F.conv2d(features, prompts, padding=pad).max(dim=1, keepdim=True)[0] / kernel_area
        correlation = F.interpolate(correlation, size=tuple(input_size))
        
        max_corre = correlation.max(dim=0, keepdim=True)[0]
        min_corre = correlation.min(dim=0, keepdim=True)[0]
        correlation = (correlation - min_corre) / (max_corre - min_corre + 1e-7)

        coordinate = self.detection(correlation)
        return coordinate
    
    @torch.no_grad()
    def fit(self, x):
        #x => [b,3,H,W]
        input_size = x.shape[-2:]
        feat_size = self.inter_rate * input_size
        feat_size = tuple(feat_size.astype(np.int32))
        
        features = self.extractor(x)

        feats_list = []
        for layer_name in self.extract_layers:
            feats_list.append(F.interpolate(features[layer_name], size=feat_size))

        #[b, C_1+C_2, H_f, W_f]
        features = torch.cat(feats_list, dim=1)
        features = F.normalize(features)

        self.prompts.append(features.detach().cpu())

    def detection(self, correlation):
        B = correlation.shape[0]
        # correlation = self.spatial_softmax(correlation)
        correlation = (correlation > self.noise_strength) * (correlation == self.maxpool(correlation)) * 1.
        for _ in range(5):
            correlation = self.gaussian_filter(correlation)
            correlation = (correlation != 0) * (correlation == self.maxpool(correlation)) * 1.0
        # torchvision.utils.save_image(correlation,
        #                     "correlation3.png",
        #                     normalize=True)

        coordinate = torch.nonzero(correlation[:, 0])
        # [batch,num,3(batch,y,x)] => [batch,num,3(batch,x,y)]
        coordinate = coordinate[:, [0, 2, 1]].detach().to("cpu")
        coordinate = [coordinate[coordinate[:, 0] == batch, 1:] for batch in range(B)]

        return coordinate


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

    def forward(self, x1, x2, x3):
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
        
        return bboxes




class Spatial_Softmax(nn.Module):
    def __init__(self, kernel_size=3, temperature=0.07, smooth=0.1, device="cuda:0"):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.smooth = smooth
        self.weights = torch.ones([1, 1, kernel_size, kernel_size]).cuda(device)

    def forward(self, x):
        softmax_ep = x - x.max()
        softmax_ep = torch.exp(softmax_ep / self.temperature)
        softmax_under = F.conv2d(softmax_ep, self.weights, padding=(self.kernel_size - 1) // 2)
        softmax_ep = softmax_ep / (softmax_under + self.smooth)
        return softmax_ep

if __name__ == "__main__":
    model = Wide_ResNet50V2_residual(threshold=0.50)
    norm = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img = cv2.imread("/mnt/kamiya/code/my_exp/zenigoke/video_images/200107_Mppicalm-k-2#18x20_001/000100.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img1 = norm(torch.from_numpy(img).float() / 255).cuda()
    
    
    img_origin = cv2.imread("/mnt/kamiya/code/my_exp/zenigoke/video_images/200107_Mppicalm-k-2#18x20_001/000101.png")
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img2 = norm(torch.from_numpy(img).float() / 255).cuda()

    
    img = cv2.imread("/mnt/kamiya/code/my_exp/zenigoke/video_images/200107_Mppicalm-k-2#18x20_001/000102.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img3 = norm(torch.from_numpy(img).float() / 255).cuda()
    
    bboxes, logits = model(img1[None], img2[None], None)
    bboxes, logits = bboxes.detach().cpu().numpy().astype(np.int32), logits.detach().cpu().numpy()

    img_det = img_origin
    for bbox, logit, in zip(bboxes, logits):
        # bbox = np.clip(bbox, a_min=0, a_max=None)
        bbox[-2:] -= bbox[:2] 
        print(bbox)
        img_det = cv2.rectangle(img_det, list(bbox), color=(0, 0, 255), thickness=5)
        img_det = cv2.putText(img_det,
                                text=f"{logit:.2%}",
                                org=(bbox[0], bbox[1]),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.,
                                color=(0, 0, 255),
                                thickness=2)
    
    cv2.imwrite("detection.png", img_det)