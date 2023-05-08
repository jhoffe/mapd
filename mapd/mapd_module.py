from abc import ABCMeta, abstractmethod

import torch
from lightning import LightningModule
from typing import Any, Optional, List
from torch import Tensor
from torch.utils.data import DataLoader


class MAPDModule(LightningModule, metaclass=ABCMeta):
    mapd_current_indices_: Optional[torch.Tensor]
    mapd_losses_: Optional[Tensor]
    mapd_indices_: Optional[Tensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapd_current_indices_ = []
        self.mapd_losses_ = None
        self.mapd_indices_ = None

    @classmethod
    @abstractmethod
    def batch_loss(self, logits: Any, y: Any) -> Tensor:
        raise NotImplemented("batch_loss method not implemented")

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        batch, indices = batch
        self.mapd_current_indices_ = indices
        return batch

    def log_batch_loss(self, logits: Any, y: Any):
        loss = self.batch_loss(logits, y)
        current_indices = self.mapd_current_indices_

        assert self.mapd_losses_ is not None

        if self.mapd_losses_ is None:
            self.mapd_losses_ = loss
            self.mapd_indices_ = current_indices
        else:
            self.mapd_losses_ = torch.cat((self.mapd_losses_, loss))
            self.mapd_indices_ = torch.cat((self.mapd_indices_, current_indices))

    def on_train_epoch_end(self) -> None:
        if self.mapd_losses_ is not None:
            self.mapd_losses_ = None
            self.mapd_indices_ = None



    def on_validation_epoch_end(self):
        # Run loss logging for probes
        #dataloader = self.probe_suite_dataloader()
        pass