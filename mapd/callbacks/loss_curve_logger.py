import os

import lightning as L
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightning.pytorch.strategies import SingleDeviceStrategy


class LossCurveLogger(L.Callback):
    def __init__(self, dir: str, wandb_suffix: str) -> None:
        super().__init__()
        self.dir = dir
        self.wandb_suffix = wandb_suffix
        self.train_loss_curves = []
        self.val_loss_curves = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        unreduced_losses = outputs["unreduced_loss"]
        indices = outputs["indices"]
        y = outputs["y"]
        y_hat = outputs["y_hat"]

        self.train_loss_curves.append(
            (
                batch_idx,
                unreduced_losses.detach(),
                indices.detach(),
                y.detach(),
                y_hat.detach(),
            )
        )

    def get_path(self, version: int) -> str:
        return os.path.join(self.dir, f"losses_v{version}.pt")

    def reset_loss_curves(self, stage: str):
        if stage == "train":
            self.train_loss_curves = []
        elif stage == "val":
            self.val_loss_curves = []

    def aggregate_and_write_loss_curves(
        self, stage: str, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        loss_curves = None
        if stage == "train":
            loss_curves = self.train_loss_curves
        elif stage == "val":
            loss_curves = self.val_loss_curves
        assert loss_curves is not None

        batch_ids = []
        indices = torch.cat([indices for _, _, indices, _, _ in loss_curves], 0).to(
            pl_module.dtype
        )
        y = torch.cat([y for _, _, _, y, _ in loss_curves], 0).to(pl_module.dtype)
        y_hat = torch.cat([y_hat for _, _, _, _, y_hat in loss_curves], 0).to(
            pl_module.dtype
        )

        for batch_idx, _, fns, _, _ in loss_curves:
            batch_ids.extend([batch_idx] * len(fns))

        batch_ids = torch.tensor(batch_ids, dtype=torch.int64)
        losses = torch.cat([lc[1] for lc in loss_curves], 0)

        if not isinstance(trainer.strategy, SingleDeviceStrategy):
            all_batch_ids = pl_module.all_gather(batch_ids)  # W x N
            all_indices = pl_module.all_gather(indices)  # W x N
            all_losses = pl_module.all_gather(losses)  # W x N
            all_y = pl_module.all_gather(y)  # W x N
            all_y_hat = pl_module.all_gather(y_hat)  # W x N

            self.reset_loss_curves(stage)

            if pl_module.local_rank != 0:
                return

            batch_ids = torch.cat(all_batch_ids, 0).cpu()
            indices = torch.cat(all_indices, 0).cpu()
            losses = torch.cat(all_losses, 0).cpu()
            y = torch.cat(all_y, 0).cpu()
            y_hat = torch.cat(all_y_hat, 0).cpu()

        self.write_to_dataset(batch_ids, indices, losses, pl_module, stage, y, y_hat)

        self.reset_loss_curves(stage)

    def write_to_dataset(self, batch_ids, indices, losses, pl_module, stage, y, y_hat):
        os.makedirs(self.dir, exist_ok=True)
        pa_batch_indices = pa.array(batch_ids, type=pa.uint16())
        pa_losses = pa.array(losses.cpu().numpy())
        pa_sample_indices = pa.array(indices.cpu().numpy(), type=pa.uint32())
        pa_y = pa.array(y.cpu().numpy(), type=pa.uint16())
        pa_y_hat = pa.array(y_hat.cpu().numpy(), type=pa.uint16())
        pa_epochs = pa.array([pl_module.current_epoch] * len(losses), type=pa.uint16())
        pa_stage = pa.array([stage] * len(losses), type=pa.string())
        pa_table = pa.table(
            [
                pa_batch_indices,
                pa_losses,
                pa_sample_indices,
                pa_y,
                pa_y_hat,
                pa_epochs,
                pa_stage,
            ],
            names=["batch_idx", "loss", "sample_index", "y", "y_hat", "epoch", "stage"],
        )
        pq.write_to_dataset(
            pa_table,
            self.dir,
            partition_cols=["epoch", "stage"],
            use_legacy_dataset=False,
        )

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.aggregate_and_write_loss_curves("train", trainer, pl_module)

    @staticmethod
    def create_loss_curve(batch_idx: int, outputs: dict):
        unreduced_losses = outputs["unreduced_loss"]
        indices = outputs["indices"]
        y = outputs["y"]
        y_hat = outputs["y_hat"]

        return (
            batch_idx,
            unreduced_losses.detach(),
            indices.detach(),
            y.detach(),
            y_hat.detach(),
        )

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs["indices"] is None:
            return

        self.val_loss_curves.append(self.create_loss_curve(batch_idx, outputs))

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.aggregate_and_write_loss_curves("val", trainer, pl_module)
