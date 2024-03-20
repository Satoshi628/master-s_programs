"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import GaussianBlur
import tqdm


import patchcore
import patchcore.sampler
LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super().__init__()
        self.device = device

    def load(
            self,
            device,
            **kwargs,
        ):
        self.init_features()
        
        def hook_t(module, input, output):
            self.backbone_features.append(output)
            
        self.resnet = models.wide_resnet101_2(pretrained=True).to(device)
        # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        # self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True) .to(device)
        self.resnet.layer2.register_forward_hook(hook_t)
        self.resnet.layer3.register_forward_hook(hook_t)
        self.device = device
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.eval()

        self.featuresampler = patchcore.sampler.DynamicApproximateGreedyCoresetSampler(0.1, device)
        self.unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=1, dilation=1)
        #scipy.ndimage.gaussian_filterと同様のフィルタサイズに設定
        #int(4*4+0.5) => 17
        self.gaussian_blur = GaussianBlur(17, 4.)

    def init_features(self):
        self.backbone_features=[]

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                input_image = image.to(torch.float).to(device=self.device, non_blocking=True)
                features.append(self._embed(input_image))
            return features
        return self._embed(data)

    @torch.no_grad()
    def _embed(self, images):
        """Returns feature embeddings for images."""
        bs = len(images)
        self.init_features()
        _ = self.resnet(images)
        features = self.backbone_features
        CHW = [f.shape[-3:] for f in features]
        C = max([chw[0] for chw in CHW])
        H = max([chw[1] for chw in CHW])
        W = max([chw[2] for chw in CHW])

        for idx, feature in enumerate(features):
            #patch_maker.patchifyの処理
            #[b,c,h,w] => [b,c*p*p,h//p*w//p]
            #TODO:avgpool(3,1,1)で代用できそう
            feature = self.unfolder(feature)
            #preprocessingの処理をやる
            feature = F.adaptive_avg_pool1d(feature.transpose(-1,-2), C).transpose(-1,-2)
            
            hw = CHW[idx][-2:]
            feature = feature.reshape(bs, C, hw[0], hw[1])
            feature = F.interpolate(
                feature,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            #[b,c,h//p,w//p] => [b,h//p,w//p,c]
            feature = feature.permute(0,2,3,1).reshape(-1, C)
            features[idx] = feature
        features = torch.stack(features, dim=1).flatten(1)
        features = F.adaptive_avg_pool1d(features, C)


        return features, [H,W]

    @torch.no_grad()
    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""

        features = []

        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                image = image.to(torch.float).to(self.device, non_blocking=True)
                
                features.append(self._embed(image)[0].detach().to("cpu").numpy())

        features = np.concatenate(features, axis=0)

        features, indices, distance = self.featuresampler.run(features)

        dist = distance/distance.max()
        x = np.arange(len(distance))/len(distance)
        center_dist = dist**2 + x**2
        idx = center_dist.argmin()

        self.features = torch.from_numpy(features).to(self.device)
        return idx/len(distance)

    @torch.no_grad()
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""

        scores = []
        masks = []
        visualize = []
        labels_gt = []
        masks_gt = []
        anomaly = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    anomaly.extend(image["anomaly"])
                    image = image["image"]
                _scores, _masks, _visualize = self._predict(image)
                for score, mask, vis in zip(_scores, _masks, _visualize):
                    scores.append(score)
                    masks.append(mask)
                    visualize.append(vis)
        return scores, masks, labels_gt, masks_gt, visualize, anomaly

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device, non_blocking=True)

        batchsize, _, H, W = images.shape
        features, feature_size = self._embed(images)

        #feature[N,D] vs memory feature[M,D]
        f_times_f = (features**2).sum(dim=-1)
        m_times_m = (self.features**2).sum(dim=-1)
        f_times_m = features.mm(self.features.T)
        score = (f_times_f[:,None] + m_times_m[None] - 2*f_times_m).min(dim=-1)[0]
        score = score.sqrt()

        image_scores = score.reshape(batchsize, -1).max(dim=-1)[0]
        masks = score.reshape(batchsize, *feature_size)
        masks = F.interpolate(masks[None], size=[H, W], mode="bilinear", align_corners=False)[0]
        masks = self.gaussian_blur(masks)

        visualize = np.zeros([batchsize, 3, H, W])
        return [score for score in image_scores.cpu().numpy()], [mask for mask in masks.cpu().numpy()], [vis for vis in visualize]

    def save_to_path(self, save_path: str) -> None:
        LOGGER.info("Saving PatchCore data.")
        #メモリー特徴量保存
        np.savez_compressed(os.path.join(save_path, "features"), self.features.cpu().numpy())

    def load_from_path(
            self,
            load_path: str,
            device: torch.device
        ) -> None:

        LOGGER.info("Loading and initializing PatchCore.")
        self.device = device
        self.features = np.load(os.path.join(load_path, "features.npz"))["arr_0"]
        self.features = torch.from_numpy(self.features).to(self.device)

