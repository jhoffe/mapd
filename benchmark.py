import pickle
import time

import numpy as np
from torch.utils.data import DataLoader

import mapd
import torchvision
from torchvision import transforms

from mapd.probes.make_probe_suites import make_probe_suites
from mapd.utils.make_dataloaders import make_dataloaders
from torch import nn
import lightning as L
from torch.nn import functional as F
from torch.optim import SGD
import torch
from torch.utils.data import random_split
from mapd.utils.wrap_dataset import wrap_dataset


# Define the neural network model
class Net(nn.Module):
    def __init__(self, num_labels: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_labels)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# Define the MAPDModule
class EMNISTModule(mapd.MAPDModule):
    def __init__(
            self,
            max_epochs: int = 10,
            lr: float = 0.05,
            momentum: float = 0.9,
            weight_decay: float = 0.0005,
            num_labels: int = 47
    ):
        super().__init__()
        self.model = Net(num_labels=num_labels)

        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

    def mapd_settings(self):
        return {
            "proxies_output_path": "mapd_proxies",
            "probes_output_path": "mapd_probes"
        }

    def forward(self, x):
        return self.model(x)

    def batch_loss(self, logits, y) -> torch.Tensor:
        return F.cross_entropy(logits, y, reduction="none")

    def batch_proxy_metric(self, logits, y) -> torch.Tensor:
        return -self.batch_loss(logits, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.batch_loss(logits, y).mean()
        self.mapd_log(logits, y)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch

        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.mapd_log(logits, y)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr
        )

        return {"optimizer": optimizer}


torch.set_float32_matmul_precision('high')

# Now handling the datasets
LABEL_COUNT = 47
EMNIST_ROOT = "data"
torchvision.datasets.EMNIST(root=EMNIST_ROOT, split="letters", download=True)

emnist_transforms = transforms.Compose([transforms.ToTensor()])
emnist_full = torchvision.datasets.EMNIST(EMNIST_ROOT, train=True, split="balanced", transform=emnist_transforms)
emnist_train, mnist_val = random_split(emnist_full, [0.8, 0.2])

# Define the dataloaders
BATCH_SIZE = 256
NUM_WORKERS = 16

NUM_PROXY_EPOCHS = 100
NUM_PROBES_EPOCH = 100

# We need to wrap the datasets in IDXDataset, to uniquely identify each sample.
idx_mnist_train = wrap_dataset(emnist_train)

proxy_train_dataloader = DataLoader(idx_mnist_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
                                    pin_memory=True)

train_dataloader = DataLoader(emnist_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True,
                              pin_memory=True)


def run_proxies():
    # Setup for training
    proxy_trainer = L.Trainer(max_epochs=NUM_PROXY_EPOCHS, accelerator="gpu", barebones=True, precision=16)
    proxy_emnist_module = EMNISTModule()
    proxy_trainer.fit(proxy_emnist_module.as_proxies(), proxy_train_dataloader)


# Run the benchmarks
N_LOOPS = 20

proxy_times = []
for i in range(N_LOOPS):
    print("Loop (Proxy)", i)
    start = time.perf_counter()
    run_proxies()
    end = time.perf_counter()
    proxy_times.append(end - start)

# We have now created the proxies needed for the probes
emnist_train_probes = make_probe_suites(idx_mnist_train, label_count=LABEL_COUNT, proxy_calculator="mapd_proxies",
                                        add_train_suite=True)

# Now we can create the dataloaders for the probes
probe_train_dataloader = DataLoader(emnist_train_probes, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True)

# Now the validation dataloaders
validation_dataloader = DataLoader(wrap_dataset(mnist_val), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                   shuffle=False,
                                   pin_memory=True)

# Now the validation dataloaders
validation_dataloader_without_idx = DataLoader(mnist_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                               shuffle=False,
                                               pin_memory=True)

validation_dataloaders = make_dataloaders([validation_dataloader], emnist_train_probes, dataloader_kwargs={
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "prefetch_factor": 4
})


def run_probes():
    # Now we can train the probes
    probe_trainer = L.Trainer(max_epochs=NUM_PROBES_EPOCH, accelerator="gpu", barebones=True, precision=16)
    probe_emnist_module = EMNISTModule()
    probe_trainer.fit(probe_emnist_module.as_probes(), train_dataloaders=probe_train_dataloader,
                      val_dataloaders=validation_dataloaders)


def run_without_mapd():
    emnist_module = EMNISTModule().disable_mapd()

    trainer = L.Trainer(max_epochs=NUM_PROBES_EPOCH, accelerator="gpu", barebones=True, precision=16)
    trainer.fit(emnist_module, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader_without_idx)


probes_times = []
for i in range(N_LOOPS):
    print("Loop (Probes)", i)
    start = time.perf_counter()
    run_probes()
    end = time.perf_counter()
    probes_times.append(end - start)

without_times = []
for i in range(N_LOOPS):
    print("Loop (Without)", i)
    start = time.perf_counter()
    run_without_mapd()
    end = time.perf_counter()
    without_times.append(end - start)

benchmarks = {
    "proxy": proxy_times,
    "probes": probes_times,
    "without": without_times
}
with open("benchmarks.pickle", "wb") as pickle_file:
    pickle.dump(benchmarks, pickle_file)

print(f"Average proxy time: {np.mean(proxy_times):.5f} +/- {np.std(proxy_times):.5f}")
print(f"Average probes time: {np.mean(probes_times):.5f} +/- {np.std(probes_times):.5f}")
print(f"Average without time: {np.mean(without_times):.5f} +/- {np.std(without_times):.5f}")
