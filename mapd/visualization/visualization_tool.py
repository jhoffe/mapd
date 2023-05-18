import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Tuple, Union, Optional


class MAPDVisualizationTool():
    def __init__(self, mapd_loss_dataset_path: os.PathLike) -> None:
        self.data = ds.dataset(mapd_loss_dataset_path, format="parquet")

    def probe_accuracy_plot(self):
        pass

    def consistently_learned_plot(self):
        pass

    def first_learned_plot(self):
        pass

    def loss_curve_plot(self):
        pass

    def violin_loss_plot(self):
        pass