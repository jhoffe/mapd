import os

import lightning as L
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


class ProxyLogger(L.Callback):
    def __init__(self, dir: str, wandb_suffix: str) -> None:
        super().__init__()
        self.dir = dir
        self.wandb_suffix = wandb_suffix
        self.proxy_samples = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        indices = outputs["indices"].detach()
        y = outputs["y"].detach()
        logits = outputs["logits"].detach()

        metrics = self.calculate_metrics(logits, y)

        self.proxy_samples.append((indices, y, metrics))

    @staticmethod
    def calculate_metrics(logits: torch.Tensor, y: torch.Tensor):
        smax = F.softmax(logits, dim=1)

        correctly_predicted = (logits.argmax(1) == y).float()
        p_L = smax[torch.arange(len(logits)), y]
        p_max = smax.max(1).values
        H = -torch.sum(smax * torch.log(smax + 1e-6), dim=1)

        return correctly_predicted, p_L, p_max, H

    def get_path(self, version: int) -> str:
        return os.path.join(self.dir, f"losses_v{version}.pt")

    def reset_proxy_samples(self):
        self.proxy_samples = []

    def aggregate_and_write_proxy_samples(self, pl_module: L.LightningModule) -> None:
        proxy_samples = self.proxy_samples

        all_y = []
        all_sample_indices = []
        all_correctly_predicted = []
        all_p_L = []
        all_p_max = []
        all_H = []

        for indices, y, (correctly_predicted, p_L, p_max, H) in proxy_samples:
            all_y.append(y)
            all_sample_indices.append(indices)
            all_correctly_predicted.append(correctly_predicted)
            all_p_L.append(p_L)
            all_p_max.append(p_max)
            all_H.append(H)

        # Concatenate all the tensors into torch tensors
        all_y = torch.cat(all_y, 0).to(pl_module.dtype)
        all_sample_indices = torch.cat(all_sample_indices, 0).to(pl_module.dtype)
        all_correctly_predicted = torch.cat(all_correctly_predicted, 0).to(
            pl_module.dtype
        )
        all_p_L = torch.cat(all_p_L, 0).to(pl_module.dtype)
        all_p_max = torch.cat(all_p_max, 0).to(pl_module.dtype)
        all_H = torch.cat(all_H, 0).to(pl_module.dtype)

        self.write_to_dataset(
            all_y,
            all_sample_indices,
            all_correctly_predicted,
            all_p_L,
            all_p_max,
            all_H,
            pl_module,
        )
        self.reset_proxy_samples()

    def write_to_dataset(
        self,
        all_y,
        all_sample_indices,
        all_correctly_predicted,
        all_p_L,
        all_p_max,
        all_H,
        pl_module,
    ):
        os.makedirs(self.dir, exist_ok=True)

        # Create pyarrow arrays
        all_y = pa.array(all_y.cpu().numpy(), type=pa.uint8())
        all_sample_indices = pa.array(
            all_sample_indices.cpu().numpy(), type=pa.uint32()
        )
        all_correctly_predicted = pa.array(
            all_correctly_predicted.cpu().numpy(), type=pa.bool_()
        )
        all_p_L = pa.array(all_p_L.cpu().numpy(), type=pa.float32())
        all_p_max = pa.array(all_p_max.cpu().numpy(), type=pa.float32())
        all_H = pa.array(all_H.cpu().numpy(), type=pa.float32())
        epochs = pa.array([pl_module.current_epoch] * len(all_y), type=pa.uint16())

        # Create a pyarrow table
        table = pa.table(
            [
                all_y,
                all_sample_indices,
                all_correctly_predicted,
                all_p_L,
                all_p_max,
                all_H,
                epochs,
            ],
            names=[
                "y",
                "sample_indices",
                "correctly_predicted",
                "p_L",
                "p_max",
                "H",
                "epoch",
            ],
        )

        # Write the table to a parquet dataset
        pq.write_to_dataset(
            table,
            self.dir,
            partition_cols=["epoch"],
            use_legacy_dataset=False,
        )

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self.aggregate_and_write_proxy_samples(pl_module)
