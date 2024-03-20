import abc
from typing import Union
from logging import getLogger
import logging

from torch.distributions.categorical import Categorical
from patchcore.sampler import BaseSampler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import lap

import numpy as np
import torch
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)



class Simulated_Annealing(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            iters = 1_000_000,
            dimension_to_project_features_to=128,
        ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    @torch.no_grad()
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
            self, features: Union[torch.Tensor, np.ndarray]
        ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        
        sample_indices = self._compute_simulated_annealing_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)


    def init_sample(self, features):
        index = np.random.choice(range(len(features)), [int(self.percentage*len(features))])
        return index
    
    def _compute_simulated_annealing_indices(self, feature):
        index = self.init_sample(feature)
        
        

        pass

    @staticmethod
    def _compute_batchwise_differences(
            matrix_a: torch.Tensor, matrix_b: torch.Tensor
        ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class Pseudo_Kmeans(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            iters = 1_000_000,
            dimension_to_project_features_to=128,
        ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    @torch.no_grad()
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    @torch.no_grad()
    def run(
            self, features: Union[torch.Tensor, np.ndarray]
        ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        
        sample_indices = self._compute_pseudo_kmeans_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)


    def init_sample(self, features):
        index = np.random.choice(range(len(features)), [int(self.percentage*len(features))])
        return index
    
    def _compute_pseudo_kmeans_indices(self, feature):
        index = self.init_sample(feature)
        
        

        pass

    @staticmethod
    def _compute_batchwise_differences(
            matrix_a: torch.Tensor, matrix_b: torch.Tensor
        ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class KNN_sampler(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            dimension_to_project_features_to=128,
        ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    @torch.no_grad()
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    @torch.no_grad()
    def run(
            self, features: Union[torch.Tensor, np.ndarray]
        ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        
        sample_indices = self._compute_KNN_sampling_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)


    
    def _compute_KNN_sampling_indices(self, feature):
        sampled_feat = feature
        weights = torch.ones(sampled_feat.shape[:1]).to(sampled_feat.device)
        
        k = int(len(feature)*self.percentage*0.1)

        original_num = len(sampled_feat)
        while len(sampled_feat) > original_num*self.percentage:
            idx = np.random.choice(len(sampled_feat), 1, replace=False).tolist()
            dist = (sampled_feat[idx] - sampled_feat)**2
            dist = dist.sum(dim=-1)
            _, topk_idx = torch.topk(dist, k)
            sum_weight = weights[topk_idx].sum(dim=0)
            center_feature = (weights[topk_idx,None]*sampled_feat[topk_idx]).sum(dim=0)/sum_weight
            
            topk_idx = topk_idx.to("cpu").tolist()
            sample_idx = [idx for idx in range(len(sampled_feat)) if not idx in topk_idx]
            sampled_feat = torch.cat([center_feature[None], sampled_feat[sample_idx]], dim=0)

            sample_idx = [idx for idx in range(len(weights)) if not idx in topk_idx]
            weights = torch.cat([sum_weight[None], weights[sample_idx]], dim=0)
            print(len(sampled_feat))
        
        idx_list = []
        for sample in sampled_feat:
            dist = ((feature-sample[None])**2).sum(dim=-1)
            idx_list.append(dist.argmin().item())
        return idx_list


class Balance_sampler(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            dimension_to_project_features_to=128,
        ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    @torch.no_grad()
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    @torch.no_grad()
    def run(
            self, features: Union[torch.Tensor, np.ndarray]
        ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        
        sample_indices = self._compute_balance_sampling_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)


    def _compute_balance_sampling_indices(self, feature):
        batch_feat = feature.split(16)
        likelihood = torch.zeros_like(feature[:,0])
        sqrt_dim = np.sqrt(self.dimension_to_project_features_to)
        for b in tqdm(batch_feat):
            dist = (b[:,None]-feature[None])**2
            dist = torch.sqrt(dist.sum(dim=-1))
            likelihood += torch.exp(-dist/sqrt_dim).sum(dim=0)

        likelihood /= likelihood.sum()
        likelihood = 1 - likelihood

        sampler = Categorical(likelihood)

        idx = sampler.sample(torch.Size([int(len(feature)*self.percentage)]))
        return idx.to("cpu").tolist()


class Farthest_Point_Sampler(BaseSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            dimension_to_project_features_to=128,
        ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    @torch.no_grad()
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    @torch.no_grad()
    def run(
            self, features: Union[torch.Tensor, np.ndarray]
        ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        
        sample_indices = self._compute_balance_sampling_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)


    def _compute_balance_sampling_indices(self, feature):
        sample_num = int(len(feature)*self.percentage)
        index = np.random.choice(range(len(feature)), [1])[0]
        sample_indices = [index]
        
        dist = ((feature[index, None] - feature)**2).sum(dim=-1)
        for _ in tqdm(range(sample_num-1)):
            dist_idx = dist.argmax().item()
            distance = ((feature[dist_idx, None] - feature)**2).sum(dim=-1)
            dist = torch.min(dist, distance)

            sample_indices.append(dist_idx)
        return sample_indices


if __name__ == "__main__":
    all_feat = np.load("/mnt/kamiya/code/Anomaly_detection/outputs/test_run2/_all_feature.npz")["arr_0"]
    # sampled_feat = np.load("/mnt/kamiya/code/Anomaly_detection/outputs/test_run2/_sampled_feature.npz")["arr_0"]
    sampler = Farthest_Point_Sampler(0.1, "cuda:0")
    sampled_feat = sampler.run(all_feat)
    print(all_feat.shape)
    print(sampled_feat.shape)
    pca = PCA(n_components=2)
    pca.fit(all_feat)
    all_xy = pca.transform(all_feat)
    sampled_xy = pca.transform(sampled_feat)

    all_feat = torch.from_numpy(all_feat)
    sampled_feat = torch.from_numpy(sampled_feat).to("cuda:0")

    batch_all_feat = all_feat.split(64)

    distance = []
    for f in tqdm(batch_all_feat):
        f = f.to("cuda:0", non_blocking=True)
        dist = (f[:,None] - sampled_feat[None])**2
        dist = torch.sqrt(dist.sum(dim=-1)).min(dim=-1)[0]
        distance.append(dist)
    
    distance = torch.cat(distance, dim=0).mean()
    print(distance)

    plt.scatter(all_xy[:,0], all_xy[:,1], s=1, c="b")
    plt.scatter(sampled_xy[:,0], sampled_xy[:,1], s=1, c="r")
    plt.title(f"mean distance:{distance.item()}")
    plt.savefig("Farthest Point Sampler.png")







