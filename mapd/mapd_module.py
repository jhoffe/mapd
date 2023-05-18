from copy import deepcopy
from abc import ABCMeta, abstractmethod

import torch
from lightning import LightningModule
from typing import Any, List, Dict
from torch import Tensor
import pyarrow.dataset as ds
import pyarrow as pa
from torch.utils.data import Dataset
import numpy as np
from lightning.pytorch.utilities.model_helpers import is_overridden


class MAPDModule(LightningModule, metaclass=ABCMeta):
    mapd_current_indices_: torch.Tensor = torch.empty(0, dtype=torch.int64)
    mapd_indices_: List[Tensor] = []

    mapd_losses_: List[Tensor] = []
    mapd_y_hats_: List[Tensor] = []
    mapd_ys_: List[Tensor] = []
    mapd_stages_: List[str] = []
    mapd_proxy_metrics_: List[Tensor] = []

    as_proxies_: bool = False
    as_probes_: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapd_current_indices_ = torch.empty(0, dtype=torch.int64)
        self.mapd_losses_ = []
        self.mapd_stages_ = []
        self.mapd_proxy_metrics_ = []
        self.mapd_indices_ = []
        self.mapd_y_hats_ = []
        self.mapd_ys_ = []
        self.as_proxies_ = False
        self.as_probes_ = False
        self.is_val_probes_ = False

        self.probes_dataset = None

        if is_overridden("on_before_batch_transfer", self, MAPDModule):
            raise ValueError("on_before_batch_transfer is a reserved method name. Please rename your method.")

    @classmethod
    @abstractmethod
    def batch_loss(cls, logits: Any, y: Any) -> Tensor:
        raise NotImplemented("batch_loss method not implemented")

    @classmethod
    @abstractmethod
    def mapd_settings(cls) -> Dict[str, Any]:
        raise NotImplemented("mapd_settings method not implemented")

    def _get_setting(self, setting_name: str, default: Any = None) -> Any:
        return self.mapd_settings().get(setting_name, default)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch, indices = batch
        self.mapd_current_indices_ = indices
        return batch

    def batch_proxy_metric(self, logits: Any, y: Any) -> Tensor:
        softmax = torch.softmax(logits, dim=1)
        # softmax confidence on correct class
        return softmax[torch.arange(softmax.shape[0]), y]

    def _mapd_log_proxy(self, logits: Any, y: Any):
        proxy_metrics = self.batch_proxy_metric(logits, y).detach()
        self.mapd_proxy_metrics_.append(proxy_metrics)

    def _mapd_log_probes(self, logits: Any, y: Any) -> None:
        batch_losses = self.batch_loss(logits, y).detach()
        self.mapd_losses_.append(batch_losses)
        self.mapd_y_hats_.append(logits.argmax(dim=1).detach())
        self.mapd_ys_.append(y.detach())
        self.mapd_stages_ += ["train" if self.training else "val"] * batch_losses.shape[0]

    def mapd_log(self, logits: Any, y: Any) -> "MAPDModule":
        if not self.trainer.sanity_checking:
            if self.as_proxies_:
                self.mapd_indices_.append(self.mapd_current_indices_)
                self._mapd_log_proxy(logits, y)
                return self

            if self.training or (not self.training and self.is_val_probes_):
                self.mapd_indices_.append(self.mapd_current_indices_)
                self._mapd_log_probes(logits, y)

        return self

    def _reset_mapd_attrs(self) -> "MAPDModule":
        self.mapd_current_indices_ = torch.empty(0, dtype=torch.int64)
        self.mapd_losses_ = []
        self.mapd_proxy_metrics_ = []
        self.mapd_indices_ = []
        self.mapd_stages_ = []
        self.mapd_y_hats_ = []
        self.mapd_ys_ = []

        return self

    def as_proxies(self) -> "MAPDModule":
        self.as_proxies_ = True
        self.as_probes_ = False

        return self

    def as_probes(self, probes_dataset: Dataset) -> "MAPDModule":
        self.as_probes_ = True
        self.as_proxies_ = False

        self.probes_dataset = deepcopy(probes_dataset)
        self.probes_dataset.only_probes = True

        return self

    def _write_proxies(self) -> None:
        sample_indices = torch.cat(self.mapd_indices_).cpu().numpy()
        sample_proxy_metrics = torch.cat(self.mapd_proxy_metrics_).cpu().numpy()
        epochs = np.full(sample_indices.shape, self.current_epoch)

        table = pa.table(
            [
                pa.array(sample_indices),
                pa.array(sample_proxy_metrics),
                pa.array(epochs),
            ],
            names=["sample_index", "proxy_metric", "epoch"],
        )

        proxies_output_path = self._get_setting("proxies_output_path", "mapd_proxies")

        ds.write_dataset(table, proxies_output_path,
                         partitioning=ds.partitioning(pa.schema([("epoch", pa.int64())]), flavor="filename"),
                         existing_data_behavior="overwrite_or_ignore", format="parquet")

    def _write_probes(self) -> None:
        sample_indices = torch.cat(self.mapd_indices_).cpu().numpy()
        sample_losses = torch.cat(self.mapd_losses_).cpu().numpy()
        sample_y_hats = torch.cat(self.mapd_y_hats_).cpu().numpy()
        sample_ys = torch.cat(self.mapd_ys_).cpu().numpy()
        epochs = np.full(sample_indices.shape, self.current_epoch)

        table = pa.table(
            [
                pa.array(sample_indices),
                pa.array(sample_losses),
                pa.array(sample_y_hats),
                pa.array(sample_ys),
                pa.array(epochs),
                pa.array(self.mapd_stages_)
            ],
            names=["sample_index", "loss", "y_hat", "y", "epoch", "stage"],
        )

        probes_output_path = self._get_setting("probes_output_path", "mapd_probes")

        ds.write_dataset(table, probes_output_path,
                         partitioning=ds.partitioning(pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
                                                      flavor="filename"),
                         existing_data_behavior="overwrite_or_ignore", format="parquet")

    def on_train_epoch_end(self) -> None:
        if self.as_proxies_:
            self._write_proxies()

        if self.as_probes_:
            self._write_probes()

        self._reset_mapd_attrs()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.as_proxies_:
            return

        self.is_val_probes_ = dataloader_idx == 0
