import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clip(
            tensor + torch.randn(tensor.size()) * self.std + self.mean,
            0.0,
            1.0,
        )


class ClampRangeTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.clamp(x, 0.0, 1.0)


class ProbeSuiteGenerator(Dataset):
    dataset: Dataset
    remaining_indices: list = []
    used_indices: list = []
    dataset_len: int
    label_count: int

    suites: Dict[int, Tuple[Tuple[torch.Tensor, int], int]] = {}
    index_to_suite: Dict[int, str] = {}

    only_probes: bool = False

    def __init__(
        self,
        dataset: Dataset,
        dataset_len: int,
        label_count: int,
        num_probes: int = 500,
        corruption_module: Optional[Union[torch.nn.Module, transforms.Compose]] = None,
        only_probes: bool = False,
    ):
        self.dataset = dataset
        assert hasattr(self.dataset, "score"), "Dataset must have a score attribute"
        self.dataset_len = dataset_len
        self.used_indices = []
        self.remaining_indices = list(range(dataset_len))
        self.label_count = label_count
        self.num_probes = num_probes

        if corruption_module is None:
            self.corruption_module = transforms.Compose(
                [
                    AddGaussianNoise(mean=0.0, std=0.1),
                    ClampRangeTransform(),
                ]
            )

        self.corruption_module = corruption_module
        self.only_probes = only_probes

        self.suites = {}
        self.index_to_suite = {}

    def generate(self):
        self.generate_atypical()
        self.generate_typical()
        self.generate_random_outputs()
        self.generate_random_inputs_outputs()
        self.generate_corrupted()

        assert len(np.intersect1d(self.remaining_indices, self.used_indices)) == 0
        assert (
            len(np.unique(self.remaining_indices + self.used_indices))
            == self.dataset_len
        )
        assert (
            len(np.unique(self.remaining_indices)) + len(np.unique(self.used_indices))
            == self.dataset_len
        )

    def add_suite(
        self, name: str, suite: List[Tuple[torch.Tensor, int, int]]
    ) -> "ProbeSuiteGenerator":
        for (sample, target), idx in suite:
            self.index_to_suite[idx] = name
            self.suites[idx] = ((sample, target), idx)

        return self

    def generate_typical(self):
        sorted_indices = np.argsort(self.dataset.score)
        subset = self.get_subset(indices=sorted_indices[: self.num_probes])
        suite = [((x, y), idx) for (x, y, _), idx in zip(subset, subset.indices)]

        self.add_suite("typical", suite)

    def generate_atypical(self):
        sorted_indices = np.argsort(self.dataset.score)
        subset = self.get_subset(
            indices=sorted_indices[-self.num_probes :]  # noqa: E203
        )  # noqa: E203
        atypical = [((x, y), idx) for (x, y, _), idx in zip(subset, subset.indices)]

        self.add_suite("atypical", atypical)

    def generate_random_outputs(self):
        subset = self.get_subset()
        suite = [
            (
                (x, random.choice([i for i in range(self.label_count) if i != y])),
                idx,
            )
            for (x, y, _), idx in zip(subset, subset.indices)
        ]

        self.add_suite("random_outputs", suite)

    def generate_random_inputs_outputs(self):
        subset = self.get_subset()

        suite = [
            ((torch.rand_like(x), torch.randint(0, self.label_count, (1,)).item()), idx)
            for (x, y, _), idx in zip(subset, subset.indices)
        ]

        self.add_suite("random_inputs_outputs", suite)

    def generate_corrupted(self):
        subset = self.get_subset()

        suite = [
            ((self.corruption_module(x), y), idx)
            for (x, y, _), idx in zip(subset, subset.indices)
        ]

        self.add_suite("corrupted", suite)

    def get_subset(
        self,
        indices: Optional[list[int]] = None,
    ) -> Subset:
        if indices is None:
            subset_indices = np.random.choice(
                self.remaining_indices, self.num_probes, replace=False
            ).tolist()
        else:
            subset_indices = indices

        self.used_indices.extend(subset_indices)
        self.remaining_indices = [
            idx for idx in self.remaining_indices if idx not in subset_indices
        ]

        return Subset(self.dataset, subset_indices)

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, int], int]:
        if self.only_probes:
            keys = list(self.suites.keys())

            return self.suites[keys[index]]

        if index in self.used_indices:
            return self.suites[index]

        sample, target, _ = self.dataset[index]

        return (sample, target), index

    def __len__(self):
        if self.only_probes:
            return len(self.index_to_suite)

        return self.dataset_len


def make_probe_suites(
    input_filepath: str,
    output_filepath: str,
    dataset: str,
    label_count: int,
    num_probes: int = 500,
    use_c_scores: bool = True,
    corruption_module: Optional[Union[torch.nn.Module, transforms.Compose]] = None,
):
    data = torch.load(
        os.path.join(
            input_filepath,
            dataset,
            "train_c_scores.pt" if use_c_scores else "train_mem_scores.pt",
        )
    )
    dataset_length = len(data)

    corruption_module = (
        corruption_module
        if corruption_module is not None
        else transforms.Compose(
            [
                AddGaussianNoise(mean=0.0, std=0.1 if "cifar" in dataset else 0.25),
                ClampRangeTransform(),
            ]
        )
    )

    probe_suite = ProbeSuiteGenerator(
        data,
        dataset_length,
        label_count,
        num_probes=num_probes,
        corruption_module=corruption_module,
    )
    probe_suite.generate()

    output_name = (
        "train_probe_suite_c_scores.pt"
        if use_c_scores
        else "train_probe_suite_mem_scores.pt"
    )

    torch.save(probe_suite, os.path.join(output_filepath, dataset, output_name))
