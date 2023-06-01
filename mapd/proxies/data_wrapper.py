from mapd.proxies.proxy_calculator import ProxyCalculator
from typing import Dict, Optional


class ProxyPreprocessor():
    def __init__(self, proxy_dataset_path: str) -> None:
        self.proxy_calculator = ProxyCalculator(proxy_dataset_path)

    def load_proxy_scores(self) -> Dict[int, float]:
        self.proxy_calculator.load()
        return self.proxy_calculator.calculate_proxy_scores("p_L")
    
    
class ScoreDataset():
    def __init__(
        self,
        proxy_dataset_path: str,
        score: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__()
        self.proxy_preprocessor = ProxyPreprocessor(proxy_dataset_path)
        if score is not None:
            self.score = score
        else:
            self.score = self.proxy_preprocessor.load_proxy_scores()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (x, y, score) where score is the proxy score of the sample
        """
        sample = super().__getitem__(index)
        if self.score is None:
            return sample

        score = self.score[index]
        return sample, score
