"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler
from patchcore.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD
LOGGER = logging.getLogger(__name__)

from patchcore.speedtester import SpeedTester
Tester = SpeedTester()

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            anomaly_score_num_nn=1,
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(False, 4),
            **kwargs,
        ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                input_image = image.to(torch.float).to(self.device)
                features.append(self._embed(input_image))
            return features
        return self._embed(data)

    @torch.no_grad()
    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        if features[0].shape[1] == features[0].shape[2]:
            features = [feat.permute(0,3,1,2) for feat in features]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    @torch.no_grad()
    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        image_size = [1] + list(input_data.dataset.imagesize)
        test_img = torch.zeros(image_size).to(self.device)
        features = self.forward_modules["feature_aggregator"](test_img)
        
        features = [features[layer] for layer in self.layers_to_extract_from]
        if features[0].shape[1] == features[0].shape[2]:
            features = [feat.permute(0,3,1,2) for feat in features]
        H = 1
        for feat in features:
            if H < feat.shape[2]:
                H = feat.shape[2]
        patch_size = image_size[2] // H


        def _image_to_features(input_image):
            input_image = input_image.to(torch.float).to(self.device)
            return self._embed(input_image, detach=True)

        features = []
        images = []
        
        in_std = np.array(IMAGENET_STD).reshape(1, -1, 1, 1)
        in_mean = np.array(IMAGENET_MEAN).reshape(1, -1, 1, 1)
        with tqdm.tqdm(input_data, desc="Computing support features...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]

                    # image = image[:1]
                
                # # 勾配計算の有効化
                # image.requires_grad = True
                # feat = _image_to_features(image)

                # # 出力の最大値に対する勾配を計算
                # self.backbone.zero_grad()
                # output_sum = feat[7*14+7].sum()
                # output_sum.backward()

                # # 勾配の取得
                # gradients = image.grad.data
                # print(gradients.shape)
                # gradients = torch.norm(gradients, dim=1)
                
                # f, axes = plt.subplots(1, 2)
                # in_std = np.array(IMAGENET_STD).reshape(-1, 1, 1)
                # in_mean = np.array(IMAGENET_MEAN).reshape(-1, 1, 1)
                # image = np.clip((image[0].cpu().detach().numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

                # axes[0].imshow(image)
                # seg = axes[1].imshow(gradients[0].cpu().detach().numpy())
                # f.colorbar(seg, ax=axes[1])
                # f.set_size_inches(3 * 2, 3)
                # f.tight_layout()
                # f.savefig("patch_grad.png")
                # plt.close()
                # input("owa")

                features.append(_image_to_features(image))
                image = np.clip(
                        (image.detach().cpu().numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)
                images.append(image)

        #images.shape=[num,3,224,224]
        images = np.concatenate(images, axis=0)
        num = images.shape[0]
        H,W = images.shape[2:4]
        H,W = H//patch_size,W//patch_size
        #images.shape=[num,3,28,28]
        features = np.concatenate(features, axis=0)
        images = np.stack(np.split(images, H, -2), axis=1)
        images = np.stack(np.split(images, W, -1), axis=2)
        images = images.reshape(features.shape[0], 3, patch_size, patch_size)

        #パッチ場所の保存
        x = np.arange(W)
        y = np.arange(H)
        x = np.tile(x[None], (H, 1))
        y = np.tile(y[:, None], (1, W))
        #xy.shape=[2,H,W]
        xy = np.stack([x, y])
        #xy.shape=[num,2,H,W]
        xy = np.tile(xy[None], (num, 1, 1, 1))
        #xy.shape=[N,2]
        xy = xy.transpose(0,2,3,1).reshape(-1, 2)

        #画像の使用頻度の保存
        N = np.arange(num)
        N = np.tile(N[:, None, None], (1, H, W)).flatten()

        features, indices, distance, change_data_dist = self.featuresampler.run(features)

        
        dist = distance/distance.max()
        x = np.arange(len(distance))/len(distance)
        center_dist = dist**2 + x**2
        idx = center_dist.argmin()
        
        # dist = (distance-distance.min())/(distance.max()-distance.min())
        # idx = np.abs(dist - 0.2).argmin()


        # 二回微分最大値
        # distance = np.gradient(distance)
        # distance = np.gradient(distance)
        # idx = distance.argmax()

        self.features = features
        # self.images = images
        # self.position = xy
        # self.use_imgidx = N
        self.images = images[indices]
        self.position = xy[indices]
        self.use_imgidx = N[indices]
        self.distance = distance
        self.anomaly_scorer.fit(detection_features=[features])
        return float(idx)/len(distance), change_data_dist

    @torch.no_grad()
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        with Tester["predict_all"]:
            """This function provides anomaly scores/maps for full dataloaders."""
            _ = self.forward_modules.eval()

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
        print(Tester)
        return scores, masks, labels_gt, masks_gt, visualize, anomaly

    def _predict(self, images):
        with Tester["predict_one"]:
            """Infer score and mask for a batch of images."""
            with Tester["get_feature"]:
                images = images.to(torch.float).to(self.device)
                _ = self.forward_modules.eval()

                batchsize, _, H, W = images.shape
                features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            with Tester["get_score"]:
                scores, _, indices = self.anomaly_scorer.predict([features])
            visualize = self.images[indices[:,0]]
            visualize = visualize.reshape(batchsize, H//8, W//8, 3, 8, 8)
            visualize = visualize.transpose(0, 3, 1, 4, 2, 5)
            visualize = visualize.reshape(batchsize, 3, H, W)
            # visualize = np.zeros([batchsize, 3, H, W])
            with Tester["calc_score"]:
                patch_scores = image_scores = scores
                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                )
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
                image_scores = self.patch_maker.score(image_scores)

                patch_scores = self.patch_maker.unpatch_scores(
                    patch_scores, batchsize=batchsize
                )
                scales = patch_shapes[0]
                patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

                masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks], [vis for vis in visualize]

    def set_feature(self, parcent):
        n = int(parcent * self.features.shape[0])
        self.anomaly_scorer.fit(detection_features=[self.features[:n]])


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
