from torch.utils.data import Dataset

from mapd.probes.utils.idx_dataset import IDXDataset


def wrap_dataset(dataset: Dataset) -> IDXDataset:
    return IDXDataset(dataset)
