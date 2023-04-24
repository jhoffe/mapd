import os
from typing import Dict

import pyarrow.parquet as pq


class ProxyCalculator:
    def __init__(self, proxy_dataset_path: os.PathLike):
        self.proxy_df = None
        self.proxy_dataset_path = proxy_dataset_path
        self.proxy_dataset = pq.ParquetDataset(proxy_dataset_path)

    def load(self, columns: list = None):
        self.proxy_df = self.proxy_dataset.read(columns=columns).to_pandas()
        self.proxy_df["epoch"] = self.proxy_df["epoch"].astype(int)

        return self.proxy_df

    def calculate_proxy_scores(self, proxy_name: str) -> Dict[int, float]:
        scores = self.proxy_df.groupby(["sample_indices"]).agg({proxy_name: "sum"})

        scores[proxy_name] = (scores[proxy_name] - scores[proxy_name].min()) / (
            scores[proxy_name].max() - scores[proxy_name].min()
        )

        return scores[proxy_name].to_dict()
