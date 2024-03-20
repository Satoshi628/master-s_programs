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
import patchcore.backbones
import patchcore.common
import patchcore.sampler
from patchcore.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD
LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super().__init__()
        self.device = device

    def load(
            self,
            device,
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(False, 4),
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

        self.featuresampler = featuresampler
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

        features, indices, distance, change_data_dist = self.featuresampler.run(features)

        dist = distance/distance.max()
        x = np.arange(len(distance))/len(distance)
        center_dist = dist**2 + x**2
        idx = center_dist.argmin()

        self.features = torch.from_numpy(features[:idx]).to(self.device)
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



    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        #パッチ内画像の保存
        np.savez_compressed(os.path.join(save_path, prepend + "patch_images"), self.images)
        np.savez_compressed(os.path.join(save_path, prepend + "distance"), self.distance)
        np.savez_compressed(os.path.join(save_path, prepend + "position"), self.position)
        np.savez_compressed(os.path.join(save_path, prepend + "use_image_idx"), self.use_imgidx)

        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
            self,
            load_path: str,
            device: torch.device,
            nn_method: patchcore.common.FaissNN(False, 4),
            prepend: str = "",
        ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")

        self.images = np.load(os.path.join(load_path, prepend + "patch_images.npz"))["arr_0"]

        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
