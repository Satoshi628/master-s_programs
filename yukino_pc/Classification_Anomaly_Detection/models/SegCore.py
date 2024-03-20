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

class ClassCore(PatchCore):
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


    @torch.no_grad()
    def _fill_memory_bank(self, input_data):
        
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

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()

        b = images.shape[0]
        _, h, w = self.feature_map_shape

        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            score, _, _ = self.anomaly_scorer.predict([features])

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

