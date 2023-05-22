import os
from typing import Dict

import pyarrow as pa
from pyarrow import dataset as ds


class ProxyCalculator:
    def __init__(self, proxy_dataset_path: os.PathLike, proxy_name: str):
        self.proxy_df = None
        self.proxy_dataset_path = proxy_dataset_path
        self.proxy_dataset = ds.dataset(
            proxy_dataset_path,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("epoch", pa.int64())]), flavor="filename"
            ),
        )
        self.proxy_name = proxy_name

    def load(self, columns: list = None):
        self.proxy_df = self.proxy_dataset.to_table(columns=columns).to_pandas()
        self.proxy_df["epoch"] = self.proxy_df["epoch"].astype(int)

        return self.proxy_df

    def calculate_proxy_scores(self) -> Dict[int, float]:
        scores = self.proxy_df.groupby(["sample_index"]).agg({self.proxy_name: "sum"})

        scores[self.proxy_name] = (
            scores[self.proxy_name] - scores[self.proxy_name].min()
        ) / (scores[self.proxy_name].max() - scores[self.proxy_name].min())

        return scores[self.proxy_name].to_dict()
