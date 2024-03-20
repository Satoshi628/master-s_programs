import abc
from typing import Union

import tqdm
import numpy as np
import torch
import torch.distributions as D


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
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
        self.mapper = None
    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        if self.mapper is None:
            self.mapper = torch.nn.Linear(
                features.shape[1], self.dimension_to_project_features_to, bias=False
            )
        _ = self.mapper.to(self.device)
        features = features.to(self.device)
        return self.mapper(features)

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
        sample_indices, distance, coreset_distance = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features), sample_indices, distance, coreset_distance

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


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            number_of_starting_points: int = 10,
            dimension_to_project_features_to: int = 128,
        ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        coreset_distance = []
        num_coreset_samples = int(len(features) * self.percentage)

        first_dist = torch.max(approximate_coreset_anchor_distances).item()

        for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
            select_distance = torch.max(approximate_coreset_anchor_distances).item()
            select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
            coreset_distance.append(select_distance)
            coreset_indices.append(select_idx)
            coreset_select_distance = self._compute_batchwise_differences(
                features, features[select_idx : select_idx + 1]  # noqa: E203
            )
            approximate_coreset_anchor_distances = torch.cat(
                [approximate_coreset_anchor_distances, coreset_select_distance],
                dim=-1,
            )
            approximate_coreset_anchor_distances = torch.min(
                approximate_coreset_anchor_distances, dim=1
            ).values.reshape(-1, 1)

        coreset_distance = np.array(coreset_distance)
        # import matplotlib.pyplot as plt
        # idx = np.arange(coreset_distance.shape[0])
        # plt.plot(idx, coreset_distance)
        # plt.savefig('distance.png')
        # plt.close()
        # # input("保存完了")
        # print("距離データ保存完了")

        return np.array(coreset_indices), coreset_distance, 0.0



class ApproximateGreedyCoresetSamplerTimes(GreedyCoresetSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            number_of_starting_points: int = 10,
            dimension_to_project_features_to: int = 128,
        ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)


    def run(
            self, features: Union[torch.Tensor, np.ndarray],
            pre_features = None,
            choice_idx=None,
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
        
        if pre_features is not None:
            pre_features = torch.from_numpy(pre_features)
            reduced_pre_features = self._reduce_features(pre_features)
        else:
            reduced_pre_features = None
        
        reduced_features = self._reduce_features(features)
        sample_indices, distance, coreset_distance = self._compute_greedy_coreset_indices(reduced_features, reduced_pre_features, choice_idx)
        
        if choice_idx is not None:
            sample_indices = np.concatenate([choice_idx, sample_indices])
        if pre_features is not None:
            features = torch.cat([features, pre_features])
        
        features = features[sample_indices]
        return self._restore_type(features), sample_indices, distance, coreset_distance


    def _compute_greedy_coreset_indices(self, features: torch.Tensor, pre_features=None, choice_idx=None) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """

        if pre_features is not None:
            #2回目の試行
            #最初の10点で平均距離計測
            approximate_distance_matrix = self._compute_batchwise_differences(pre_features, pre_features[choice_idx[:10]])
            approximate_coreset_anchor_distances = approximate_distance_matrix.mean(dim=-1).reshape(-1, 1)
            #それ以外の点で最小距離計測
            approximate_distance_matrix = self._compute_batchwise_differences(pre_features, pre_features[choice_idx[10:]])
            approximate_coreset_anchor_distances = torch.cat([approximate_coreset_anchor_distances, approximate_distance_matrix],dim=-1)
            approximate_coreset_anchor_distances = torch.min(approximate_coreset_anchor_distances, dim=1).values.reshape(-1, 1)
            first_dist = torch.max(approximate_coreset_anchor_distances).item()

            features = torch.cat([features, pre_features],dim=0)
            #最初の10点で平均距離計測
            approximate_distance_matrix = self._compute_batchwise_differences(features, pre_features[choice_idx[:10]])
            approximate_coreset_anchor_distances = approximate_distance_matrix.mean(dim=-1).reshape(-1, 1)
            #それ以外の点で最小距離計測
            approximate_distance_matrix = self._compute_batchwise_differences(features, pre_features[choice_idx[10:]])
            approximate_coreset_anchor_distances = torch.cat([approximate_coreset_anchor_distances, approximate_distance_matrix],dim=-1)
            approximate_coreset_anchor_distances = torch.min(approximate_coreset_anchor_distances, dim=1).values.reshape(-1, 1)
            change_data_distance = torch.max(approximate_coreset_anchor_distances).item()
        else:
            number_of_starting_points = np.clip(self.number_of_starting_points, None, len(features))
            start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()
            approximate_distance_matrix = self._compute_batchwise_differences(features, features[start_points])
            approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)
            first_dist = 0.
            change_data_distance = 0.
        
        coreset_indices = []
        coreset_distance = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
            select_distance = torch.max(approximate_coreset_anchor_distances).item()
            select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
            coreset_distance.append(select_distance)
            coreset_indices.append(select_idx)
            coreset_select_distance = self._compute_batchwise_differences(
                features, features[select_idx : select_idx + 1]  # noqa: E203
            )
            approximate_coreset_anchor_distances = torch.cat(
                [approximate_coreset_anchor_distances, coreset_select_distance],
                dim=-1,
            )
            approximate_coreset_anchor_distances = torch.min(
                approximate_coreset_anchor_distances, dim=1
            ).values.reshape(-1, 1)

        coreset_distance = np.array(coreset_distance)

        return np.array(coreset_indices), coreset_distance, change_data_distance - first_dist


class FewShotApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
            self,
            percentage: float,
            device: torch.device,
            number_of_starting_points: int = 10,
            dimension_to_project_features_to: int = 128,
        ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)


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
        
        features = self._increace_features(features)

        reduced_features = self._reduce_features(features)

        sample_indices, distance = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features), sample_indices, distance


    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        coreset_distance = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
            select_distance = torch.max(approximate_coreset_anchor_distances).item()
            select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
            coreset_distance.append(select_distance)
            coreset_indices.append(select_idx)
            coreset_select_distance = self._compute_batchwise_differences(
                features, features[select_idx : select_idx + 1]  # noqa: E203
            )
            approximate_coreset_anchor_distances = torch.cat(
                [approximate_coreset_anchor_distances, coreset_select_distance],
                dim=-1,
            )
            approximate_coreset_anchor_distances = torch.min(
                approximate_coreset_anchor_distances, dim=1
            ).values.reshape(-1, 1)

        coreset_distance = np.array(coreset_distance)
        # import matplotlib.pyplot as plt
        # idx = np.arange(coreset_distance.shape[0])
        # plt.plot(idx, coreset_distance)
        # plt.savefig('distance.png')
        # plt.close()
        # # input("保存完了")
        # print("距離データ保存完了")

        return np.array(coreset_indices), coreset_distance

    def _increace_features(self, features: torch.Tensor) -> torch.Tensor:
        #カーネル密度推定
        sigma = torch.std(features, dim=0, keepdim=True)
        #silverman's-ruleでバンド幅hを決める
        h = (3/4*features.shape[0])**(-0.2)*sigma
        h = torch.clip(h, 1e-6, None)
        h = h.repeat(features.shape[0], 1)
        normal = D.Normal(loc=features, scale=h)
        distribution = D.Independent(normal, 1)
        sample_num = 200*1000//features.shape[0]

        return distribution.sample(torch.Size([sample_num])).flatten(0,1)

class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]
