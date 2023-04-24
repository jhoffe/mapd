from typing import Dict

from torch.utils.data import Dataset


class ProxyDataset(Dataset):
    def __init__(self, dataset: Dataset, proxy_scores: Dict[int, float]):
        self.dataset = dataset
        self.proxy_scores = proxy_scores

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        score = self.proxy_scores[index]

        return sample, target, 1 - score
