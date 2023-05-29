from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from lightning import LightningModule
from lightning.pytorch.utilities.model_helpers import is_overridden
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from mapd.classifiers.make_mapd_classifier import make_mapd_classifier
from mapd.classifiers.make_predictions import make_predictions
from mapd.probes.make_probe_suites import make_probe_suites
from mapd.probes.probe_suite_generator import ProbeSuiteDataset
from mapd.probes.utils.idx_dataset import IDXDataset
from mapd.visualization.visualization_tool import MAPDVisualizationTool


class MAPDModule(LightningModule, metaclass=ABCMeta):
    mapd_current_indices_: torch.Tensor = torch.empty(0, dtype=torch.int64)
    mapd_indices_: List[Tensor] = []

    mapd_losses_: List[Tensor] = []
    mapd_y_hats_: List[Tensor] = []
    mapd_ys_: List[Tensor] = []
    mapd_stages_: List[str] = []
    mapd_proxy_metrics_: List[Tensor] = []
    mapd_probe_metrics_ = []

    as_proxies_: bool = False
    as_probes_: bool = False
    mapd_disabled_: bool = False

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
        self.mapd_disabled_ = False
        self.is_val_probes_ = False
        self.mapd_probe_metrics_ = []

        self.probes_dataset = None

        if is_overridden("on_before_batch_transfer", self, MAPDModule):
            raise ValueError(
                "on_before_batch_transfer is a reserved method name. \
                Please rename your method."
            )

    @classmethod
    @abstractmethod
    def batch_loss(cls, logits: Any, y: Any) -> Tensor:
        raise NotImplementedError("batch_loss method not implemented")

    @classmethod
    @abstractmethod
    def mapd_settings(cls) -> Dict[str, Any]:
        raise NotImplementedError("mapd_settings method not implemented")

    def _get_setting(self, setting_name: str, default: Any = None) -> Any:
        return self.mapd_settings().get(setting_name, default)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.mapd_disabled_:
            return batch

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
        logits, y = logits.detach(), y.detach()
        batch_losses = self.batch_loss(logits, y).detach()

        batch = (
            batch_losses,
            logits.argmax(dim=1),
            y,
            ["train" if self.training else "val"] * batch_losses.shape[0],
        )
        self.mapd_probe_metrics_.append(batch)

    def mapd_log(self, logits: Any, y: Any) -> "MAPDModule":
        if self.mapd_disabled_:
            return self

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
        self.mapd_probe_metrics_ = []

        return self

    def as_proxies(self) -> "MAPDModule":
        self.as_proxies_ = True
        self.as_probes_ = False

        return self

    def as_probes(self) -> "MAPDModule":
        self.as_probes_ = True
        self.as_proxies_ = False

        return self

    def disable_mapd(self) -> "MAPDModule":
        self.mapd_disabled_ = True

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

        ds.write_dataset(
            table,
            proxies_output_path,
            partitioning=ds.partitioning(
                pa.schema([("epoch", pa.int64())]), flavor="filename"
            ),
            existing_data_behavior="overwrite_or_ignore",
            format="parquet",
        )

    def _write_probes(self) -> None:
        # sample_indices = torch.cat(self.mapd_indices_).cpu().numpy()
        # sample_losses = torch.cat(self.mapd_losses_).cpu().numpy()
        # sample_y_hats = torch.cat(self.mapd_y_hats_).cpu().numpy()
        # sample_ys = torch.cat(self.mapd_ys_).cpu().numpy()
        # epochs = np.full(sample_indices.shape, self.current_epoch)
        mapd_losses, mapd_y_hats, mapd_ys, mapd_stages = list(
            zip(*self.mapd_probe_metrics_)
        )

        sample_indices = torch.cat(self.mapd_indices_).cpu().numpy()
        sample_losses = torch.cat(mapd_losses).cpu().numpy()
        sample_y_hats = torch.cat(mapd_y_hats).cpu().numpy()
        sample_ys = torch.cat(mapd_ys).cpu().numpy()
        sample_stages = np.concatenate(mapd_stages)

        epochs = np.full(sample_indices.shape, self.current_epoch)

        table = pa.table(
            [
                pa.array(sample_indices),
                pa.array(sample_losses),
                pa.array(sample_y_hats),
                pa.array(sample_ys),
                pa.array(epochs),
                pa.array(sample_stages),
            ],
            names=["sample_index", "loss", "y_hat", "y", "epoch", "stage"],
        )

        probes_output_path = self._get_setting("probes_output_path", "mapd_probes")

        ds.write_dataset(
            table,
            probes_output_path,
            partitioning=ds.partitioning(
                pa.schema([("epoch", pa.int64()), ("stage", pa.string())]),
                flavor="filename",
            ),
            existing_data_behavior="overwrite_or_ignore",
            format="parquet",
        )

    def on_train_epoch_end(self) -> None:
        if self.mapd_disabled_:
            return

        if self.as_proxies_:
            self._write_proxies()

        if self.as_probes_:
            self._write_probes()

        self._reset_mapd_attrs()

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.mapd_disabled_:
            return

        if self.as_proxies_:
            return

        self.is_val_probes_ = dataloader_idx == 0

    def make_probe_suites(
        self,
        dataset: IDXDataset,
        num_labels: int,
        num_probes: int = 500,
        add_train_suite: bool = False,
    ):
        proxies_path = self._get_setting("proxies_output_path", "mapd_proxies")

        return make_probe_suites(
            dataset,
            num_labels,
            proxies_path,
            num_probes=num_probes,
            add_train_suite=add_train_suite,
        )

    def make_mapd_classifier(
        self,
        probe_suite_dataset: ProbeSuiteDataset,
        clf: Union[str, BaseEstimator] = "xgboost",
        clf_kwargs: Optional[Dict[str, Any]] = None,
        epoch_range: Optional[Tuple[int, int]] = None,
        plot_calibration_curves: bool = False,
    ):
        probes_path = self._get_setting("probes_output_path", "mapd_probes")

        return make_mapd_classifier(
            probes_path,
            probe_suite_dataset,
            clf,
            clf_kwargs=clf_kwargs,
            epoch_range=epoch_range,
            plot_calibration_curves=plot_calibration_curves,
        )

    def mapd_predict(
        self,
        clf: BaseEstimator,
        label_encoder: LabelEncoder,
        epoch_range: Optional[Tuple[int, int]] = None,
        n_jobs: int = 1,
    ):
        probes_path = self._get_setting("probes_output_path", "mapd_probes")

        return make_predictions(
            probes_path, clf, label_encoder, epoch_range=epoch_range, n_jobs=n_jobs
        )

    def visualiaztion_tool(
        self, probe_suite_dataset: ProbeSuiteDataset
    ) -> MAPDVisualizationTool:
        probes_path = self._get_setting("probes_output_path", "mapd_probes")
        return MAPDVisualizationTool(probes_path, probe_suite_dataset)
