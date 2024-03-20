#coding: utf-8
#----- 標準ライブラリ -----#
import copy
import sys
import logging
from logging import getLogger


#----- 専用ライブラリ -----#
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

#----- 自作モジュール -----#
#from models.utils import _BACKBONES, LastLayerToExtractReachedException
from utils import _BACKBONES, LastLayerToExtractReachedException
from sampler import ApproximateGreedyCoresetSampler
from kNN import FaissNN
# from models.sampler import ApproximateGreedyCoresetSampler
# from models.kNN import FaissNN
# None


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
#logger.info('message')


class ForwardHook():
    def __init__(self, hook_dict, layer_name, last_layer_to_extract):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        # Once the last layer is executed, no more layers are executed.
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class Puzzle_net(nn.Module):
    def __init__(self, backbone_name, image_size, Near_protocol=4, vector_dim=1024, layer_names=["layer2", "layer3"], device="cuda:0"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.define_backbone(backbone_name, layer_names)
        self.backbone.to(device)

        self.patch_unfold = nn.Unfold(3, padding=1)

        layer_output_shape = self.get_feature_shape(image_size)
        print(layer_output_shape)
        self.Pre_processes = {key: Preprocessing(layer_output_shape[layer_names[-1]][0], layer_output_shape[layer_names[0]][1:]).to(device)
            for key in layer_output_shape.keys()}

        self.featuresampler = ApproximateGreedyCoresetSampler(percentage=0.1,device="cpu")
        self.kNN = FaissNN()

    def reset(self):
        # clearでなければアドレスが変更されてしまう
        # self.outputs = {}ではだめ
        self.outputs.clear()

    def get_feature_shape(self, image_size):
        self.reset()
        __inputs = torch.rand([1, 3, *image_size]).to(self.device)
        __outputs = self.feature_extraction_process(__inputs)
        return {key: out.shape[1:] for key, out in __outputs.items()}

    def feature_extraction_process(self, images):
        self.reset()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
            output = copy.deepcopy(self.outputs)
        return output

    def define_backbone(self, backbone_name, layer_names):
        self.backbone = eval(_BACKBONES[backbone_name])

        if not hasattr(self.backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layer_names:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layer_names[-1])
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = self.backbone.__dict__[
                    "_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__[
                        "_modules"][extract_idx]
            else:
                network_layer = self.backbone.__dict__[
                    "_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook))

    @torch.no_grad()
    def fit(self, dataloader):
        data_features = []
        data_feat_append = data_features.append

        logger.info("Start data fit")
        #for images in tqdm(dataloader):
        feateure_dict = self.feature_extraction_process(images)
        features = []
        for key, f in feateure_dict.items():
            features.append(self.Pre_processes[key](f))

        features = torch.cat(features, dim=1)
        # features = self.near_4_process(features)
        data_feat_append(features.to("cpu").numpy())
        
        features = np.concatenate(data_features, axis=0)
        print(features.shape)
        features = self.featuresampler.run(features)
        print(features.shape)
        self.kNN()

    def near_4_process(self, feature):
        B, C, H, W = feature.shape
        f = self.patch_unfold(feature)
        f = f.reshape(B, C, 9, -1)
        #center, top, right, bottom, left
        f = f[:, :, [4, 1, 5, 7, 3]]
        #[B,C,9,L] => [B*L,C,9]
        f = f.permute(0,3,1,2).flatten(0, 1)
        return f

    def near_8_process(self, feature):
        B, C, H, W = feature.shape
        f = self.patch_unfold(feature)
        f = f.reshape(B, C, 9, -1)
        #center, top, top-right, right, bottom-right, bottom, bottom-left, left, top-left
        f = f[:, :, [4, 1, 2, 5, 8, 7, 6, 3, 0]]
        #[B,C,9,L] => [B*L,C,9]
        f = f.permute(0,3,1,2).flatten(0, 1)
        return f

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs


class Preprocessing(nn.Module):
    def __init__(self, outputs_dim, output_img_size):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(outputs_dim)
        self.output_img_size = output_img_size

    def forward(self, feature):
        feature = F.interpolate(
            feature, size=self.output_img_size, mode="bilinear", align_corners=False)
        B, C, H, W = feature.shape
        feature = feature.flatten(2)
        # feature.shape => [batch, dim, L (H*W)]
        feature = feature.transpose(2, 1)
        feature = self.avg_pool(feature)
        feature = feature.transpose(2, 1)
        feature = feature.reshape(B, -1, H, W)
        return feature


if __name__ == "__main__":
    net = Puzzle_net("wideresnet50", [224, 224]).cuda()
    img = torch.rand([8, 3, 224, 224]).cuda()
    dicts = net.fit(img)
