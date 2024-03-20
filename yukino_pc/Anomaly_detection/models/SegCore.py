#coding: utf-8
#----- 標準ライブラリ -----#
import random
import copy
import sys
import logging
from logging import getLogger

#----- 専用ライブラリ -----#
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import GaussianBlur
import patchcore.backbones
import patchcore.common
import patchcore.sampler
from patchcore.patchcore import PatchCore
import matplotlib.pyplot as plt

#----- 自作モジュール -----#
#from models.utils import _BACKBONES, LastLayerToExtractReachedException
from models.utils import _BACKBONES
# from models.sampler import Greedy_Sampler
# from models.kNN import FaissNN


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)
#logger.info('message')

class _SegCore(nn.Module):
    def __init__(self, backbone_name, image_size, batch_size=8, vector_dim=1024, layer_names=["layer2", "layer3"], device="cuda:0"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.define_backbone(backbone_name, layer_names)
        self.backbone.to(device)

        self.patch_unfold = nn.Unfold(3, padding=1)

        layer_output_shape = self.get_feature_shape(image_size)
        print(layer_output_shape)
        self.Pre_processes = {key: Preprocessing(layer_output_shape[layer_names[-1]][0], layer_output_shape[layer_names[0]][1:]).to(device)
            for key in layer_output_shape.keys()}
        
        self.channel_pooling = Aggregator(vector_dim)

        self.featuresampler = Greedy_Sampler(percentage=0.1, device=device)
        self.segmentation_filter = Segmentor(smoothing=4, target_size=224).to(device)
        self.kNN = FaissNN()

    def reset(self):
        # clearでなければアドレスが変更されてしまう
        # self.outputs = {}ではだめ
        self.outputs.clear()

    def reset_similar_map(self):
        self.similar_map = np.zeros([0,0,0])

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
                network_layer = self.backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = self.backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(network_layer[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))

    @torch.no_grad()
    def fit(self, dataloader):
        self.reset_similar_map()
        data_features = []
        data_feat_append = data_features.append

        logger.info("Start data fit")
        for images, _, _ in tqdm(dataloader):
            images = images.cuda(self.device, non_blocking=True)

            feateure_dict = self.feature_extraction_process(images)
            features = []
            for key, f in feateure_dict.items():
                features.append(self.Pre_processes[key](f))

            features = torch.cat(features, dim=1)
            features = self.channel_pooling(features)
            # features = self.near_4_process(features)
            data_feat_append(features.to("cpu").numpy())
        
        features = np.concatenate(data_features, axis=0)

        logger.info("Start sampling")
        sample_features = self.featuresampler.run(features.transpose(0,2,3,1).reshape(-1, features.shape[1]))
        logger.info(f"sample result: {features.shape[0]*features.shape[2]*features.shape[3]} => {sample_features.shape[0]}")

        logger.info("Start saving")
        self.kNN.fit(sample_features)

        # make Seg map
        similar_map = np.zeros([sample_features.shape[0], *features.shape[-2:]]).reshape(sample_features.shape[0], -1)

        logger.info("Start make probability map")
        for feature in tqdm(features):
            values, index = self.kNN.run(10, feature.transpose(1,2,0).reshape(-1, features.shape[1]).copy(order='C'))
            values = np.exp(-values/np.sqrt(sample_features.shape[-1]))
            for v, i in zip(values.transpose(1, 0), index.transpose(1, 0)):
                similar_map[i] += v

        similar_map = np.exp(similar_map/np.sqrt(features.shape[0]))
        similar_map = similar_map / (np.sum(similar_map, axis=-1, keepdims=True) + 1e-7)
        similar_map /= similar_map.max(axis=-1)[..., None]

        self.similar_map = similar_map.reshape(sample_features.shape[0], *features.shape[-2:])

    @torch.no_grad()
    def predict(self, images):
        # logger.info("Start predict")
        feateure_dict = self.feature_extraction_process(images)
        features = []
        for key, f in feateure_dict.items():
            features.append(self.Pre_processes[key](f))

        features = torch.cat(features, dim=1)
        features = self.channel_pooling(features)
        
        features = features.to("cpu").numpy()
        
        b, c, h, w = features.shape
        
        index_H = np.tile(np.arange(h)[None, :, None], (b, 1, w)).reshape(-1)
        index_W = np.tile(np.arange(w)[None, None], (b, h, 1)).reshape(-1)

        distance, index_S = self.kNN.run(1, features.transpose(0, 2, 3, 1).reshape(b * h * w, -1))
        dist_score = distance.mean(axis=-1)

        point_socre = self.similar_map[index_S[:, 0], index_H, index_W]

        image_level_score = torch.from_numpy(dist_score.reshape(b,-1).max(axis=-1)).to(self.device)

        dist_score = torch.from_numpy(dist_score.reshape(b, 1, h, w)).to(self.device)
        point_socre = torch.from_numpy(point_socre.reshape(b, 1, h, w)).to(self.device)

        dist_score = self.segmentation_filter(dist_score)[:, 0]
        point_socre = self.segmentation_filter(point_socre)[:, 0]

        b, h, w = dist_score.shape
        dist_score = dist_score.flatten(1)
        point_socre = point_socre.flatten(1)
        dist_score = (dist_score - dist_score.min(dim=-1, keepdim=True)[0]) / (dist_score.max(dim=-1, keepdim=True)[0] - dist_score.min(dim=-1, keepdim=True)[0])
        point_socre = (point_socre - point_socre.min(dim=-1, keepdim=True)[0]) / (point_socre.max(dim=-1, keepdim=True)[0] - point_socre.min(dim=-1, keepdim=True)[0])
        
        dist_score = dist_score.view(b, h, w)
        point_socre = point_socre.view(b, h, w)

        return image_level_score, dist_score, point_socre


class SegCore(PatchCore):
    def __init__(self, device):
        super().__init__(device)

    def load(
            self,
            backbone_name,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            anomaly_score_num_nn=1,
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(True, 4),
            **kwargs,
        ):
        backbone = eval(_BACKBONES[backbone_name])
        super().load(
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=patchsize,
            patchstride=patchstride,
            anomaly_score_num_nn=anomaly_score_num_nn,
            featuresampler=featuresampler,
            nn_method=nn_method,
            **kwargs,)

        inputs = torch.rand(*([1] + input_shape), device=device)

        self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](inputs)

        self.feature_map_shape = list(features[self.layers_to_extract_from[0]].shape[-2:])
        self.feature_map_shape = [target_embed_dimension] + self.feature_map_shape

    def reset_similar_map(self):
        self.similar_map = np.zeros([0,0,0])


    @torch.no_grad()
    def _fill_memory_bank(self, input_data):
        self.reset_similar_map()
        
        """Computes and sets the support features for SPADE."""
        self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image, _, _ in data_iterator:
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        c, h, w = self.feature_map_shape
        feature_maps = features.reshape(-1, h, w, c)
        np.savez("_all_feature", features)
        batch_all_feat = torch.from_numpy(features).split(16)
        features = self.featuresampler.run(features)
        np.savez("_sampled_feature", features)
        
        sampled_feat = torch.from_numpy(features).to("cuda:0")
        distance = []
        distance_append = distance.append
        for f in tqdm.tqdm(batch_all_feat):
            f = f.to("cuda:0", non_blocking=True)
            dist = (f[:,None] - sampled_feat[None])**2
            dist = torch.sqrt(dist.sum(dim=-1)).min(dim=-1)[0]
            distance_append(dist.to("cpu"))
        
        distance = torch.cat(distance, dim=0).mean()
        logger.info(f"mean distance:{distance.item()}")

        self.anomaly_scorer.fit(detection_features=[features])

        # make Seg map
        # similar_map = np.zeros([features.shape[0], h, w]).reshape(features.shape[0], -1)


        # for feature in tqdm.tqdm(feature_maps, desc="Computing probability_map...", position=1, leave=False):
        #     means, values, index = self.anomaly_scorer.predict([feature.reshape(-1, c)])
        #     # values = np.exp((values.min()-values)/np.sqrt(c))
        #     for v, i in zip(values.transpose(1, 0), index.transpose(1, 0)):
        #         similar_map[i] += v

        # plt.clf()
        # hist_log = np.log10(np.clip(similar_map.sum(axis=-1), 1, None))
        # plt.bar(np.arange(0, similar_map.shape[0])+1, hist_log, width=1.0)
        # plt.savefig(f"hist_{input_data.dataset.product}.png")

        # similar_map = np.exp(similar_map/np.sqrt(feature_maps.shape[0]))
        # similar_map = similar_map / np.clip(np.sum(similar_map, axis=-1, keepdims=True),1e-7,None)

        # self.similar_map = similar_map.reshape(features.shape[0], h, w)

        
        # for feature in tqdm.tqdm(feature_maps, desc="Computing probability_map...", position=1, leave=False):
        #     means, values, index = self.anomaly_scorer.predict([feature.reshape(-1, c)])
        #     values = np.exp((values.min()-values)/np.sqrt(c))
        #     for v, i in zip(values.transpose(1, 0), index.transpose(1, 0)):
        #         similar_map[i] += v

        # similar_map = np.exp(similar_map-similar_map.max())
        # similar_map = similar_map / np.clip(np.sum(similar_map, axis=-1, keepdims=True),1e-7,None)

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()

        b = images.shape[0]
        _, h, w = self.feature_map_shape

        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            score, _, index_S = self.anomaly_scorer.predict([features])

            # index_H = np.tile(np.arange(h)[None, :, None], (b, 1, w)).reshape(-1)
            # index_W = np.tile(np.arange(w)[None, None], (b, h, 1)).reshape(-1)

            # point_socre = self.similar_map[index_S[:, 0], index_H, index_W]

            #patch_scores = image_scores = np.sqrt(score * (1 - point_socre))
            patch_scores = image_scores = score

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=b)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=b)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(b, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    def get_similar_map(self):
        return self.similar_map

