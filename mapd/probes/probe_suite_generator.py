from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from mapd.proxies.proxy_calculator import ProxyCalculator


def _get(x: Any) -> Any:
    return x[1]


def identity(x, y, num_labels):
    return x, y


def random_outputs(x, y, num_labels):
    return x, torch.randint(0, num_labels, (1,)).item()


def random_inputs_outputs(x, y, num_labels):
    return torch.randn_like(x), torch.randint(0, num_labels, (1,)).item()


class ProbeSuiteDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        label_count: int,
        proxy_calculator: ProxyCalculator,
        num_probes: int = 500,
        corruption_module: Optional[Union[torch.nn.Module, transforms.Compose]] = None,
        only_probes: bool = False,
        add_train_suite: bool = False,
    ):
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        self.used_indices = list()
        self.remaining_indices = list(range(self.dataset_len))
        self.label_count = label_count
        self.num_probes = num_probes

        self.corruption_module = corruption_module
        self.only_probes = only_probes

        self.proxy_calculator = proxy_calculator
        self.scores = proxy_calculator.calculate_proxy_scores()
        self.sorted_indices = list(
            dict(sorted(self.scores.items(), key=_get, reverse=True)).keys()
        )
        self.add_train_suite = add_train_suite

        assert len(self.scores) == self.dataset_len

        self.index_to_suite = {}
        self.index_to_func = {}

        self.cache = {}

    def generate(self):
        self.generate_atypical()
        self.generate_typical()
        self.generate_random_outputs()
        self.generate_random_inputs_outputs()
        if self.add_train_suite:
            self.generate_train()
        if self.corruption_module is not None:
            self.generate_corrupted()

        assert len(np.intersect1d(self.remaining_indices, self.used_indices)) == 0
        assert (
            len(np.unique(list(self.remaining_indices) + list(self.used_indices)))
            == self.dataset_len
        )
        assert (
            len(self.remaining_indices) + len(self.used_indices) == self.dataset_len
        ), f"{len(self.remaining_indices)}+{len(self.used_indices)}!={self.dataset_len}"
        assert self.dataset is not None

    def add_suite(
        self, name: str, suite_indices: Sequence[int], transform_func: callable
    ) -> "ProbeSuiteDataset":
        for idx in suite_indices:
            self.index_to_suite[idx] = name
            self.index_to_func[idx] = transform_func
            self.used_indices.append(idx)
            self.remaining_indices.remove(idx)

        return self

    def generate_train(self):
        subset = self.get_subset()
        self.add_suite("train", subset, identity)

    def generate_typical(self):
        self.add_suite("typical", self.sorted_indices[: self.num_probes], identity)

    def generate_atypical(self):
        self.add_suite("atypical", self.sorted_indices[-self.num_probes:], identity)

    def generate_random_outputs(self):
        subset = self.get_subset()
        self.add_suite("random_outputs", subset, random_outputs)

    def generate_random_inputs_outputs(self):
        subset = self.get_subset()
        self.add_suite("random_inputs_outputs", subset, random_inputs_outputs)

    def generate_corrupted(self):
        subset = self.get_subset()
        self.add_suite("corrupted", subset, self.corrupted)

    def corrupted(self, x, y):
        return self.corruption_module(x), y

    def get_subset(
        self,
        indices: Optional[list[int]] = None,
    ) -> Sequence[int]:
        if indices is None:
            subset_indices = np.random.choice(
                list(self.remaining_indices), self.num_probes, replace=False
            ).tolist()
        else:
            subset_indices = indices

        return subset_indices

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, int], int]:
        if self.only_probes:
            (x, y), idx = self.dataset[self.used_indices[index]]

            if idx in self.cache:
                return self.cache[idx], idx

            (x, y) = self.index_to_func[idx](x, y, self.label_count)
            self.cache[idx] = (x, y)

            return (x, y), idx

        (x, y), idx = self.dataset[index]

        if index in self.used_indices:
            return self.index_to_func[idx](x, y, self.label_count), idx

        return (x, y), idx

    def __len__(self):
        if self.only_probes:
            return len(self.index_to_suite)

        return self.dataset_len
