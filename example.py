from string import digits

import lightning as L
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import transforms

import mapd
from mapd.utils.make_dataloaders import make_dataloaders
from mapd.utils.wrap_dataset import wrap_dataset
from mapd.visualization.surface_predictions import display_surface_predictions


# Define the neural network model
class Net(nn.Module):
    def __init__(self, num_labels: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(80, 160, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2560, 1280)
        self.fc2 = nn.Linear(1280, num_labels)

    def forward(self, x):
        x = F.relu(self.max_pool1((self.conv1(x))))
        x = F.relu(self.conv2_drop(self.max_pool2(self.conv2(x))))
        x = x.view(-1, 2560)
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
        num_labels: int = 47,
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
            "probes_output_path": "mapd_probes",
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
        optimizer = SGD(self.parameters(), lr=self.lr)

        return {"optimizer": optimizer}


L.seed_everything(42)

torch.set_float32_matmul_precision("high")

# Now handling the datasets
LABEL_COUNT = 47
EMNIST_ROOT = "data"
torchvision.datasets.EMNIST(root=EMNIST_ROOT, split="letters", download=True)

emnist_transforms = transforms.Compose([transforms.ToTensor()])
mnist_full = torchvision.datasets.EMNIST(
    EMNIST_ROOT, train=True, split="balanced", transform=emnist_transforms
)
mnist_train, mnist_val = random_split(mnist_full, [0.8, 0.2])

# Define the dataloaders
BATCH_SIZE = 512
NUM_WORKERS = 16
model_summary = summary(
    Net(num_labels=47),
    (BATCH_SIZE, 1, 28, 28),
    verbose=2,
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
)

NUM_PROXY_EPOCHS = 50
NUM_PROBES_EPOCH = 100

# We need to wrap the datasets in IDXDataset, to uniquely identify each sample.
idx_mnist_train = wrap_dataset(mnist_train)

proxy_train_dataloader = DataLoader(
    idx_mnist_train,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
)

# Setup for training
proxy_trainer = L.Trainer(max_epochs=NUM_PROXY_EPOCHS, accelerator="gpu")
proxy_emnist_module = EMNISTModule()
proxy_trainer.fit(proxy_emnist_module.as_proxies(), proxy_train_dataloader)

# We have now created the proxies needed for the probes
emnist_train_probes = proxy_emnist_module.make_probe_suites(
    idx_mnist_train, num_labels=LABEL_COUNT, add_train_suite=True
)

# Now we can create the dataloaders for the probes
probe_train_dataloader = DataLoader(
    emnist_train_probes,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    pin_memory=True,
)

# Now the validation dataloaders
validation_dataloader = DataLoader(
    wrap_dataset(mnist_val),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True,
)

validation_dataloaders = make_dataloaders(
    [validation_dataloader],
    emnist_train_probes,
    dataloader_kwargs={
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": 4,
    },
)

# Now we can train the probes
probe_trainer = L.Trainer(max_epochs=NUM_PROBES_EPOCH, accelerator="gpu")
probe_emnist_module = EMNISTModule()
probe_trainer.fit(
    probe_emnist_module.as_probes(),
    train_dataloaders=probe_train_dataloader,
    val_dataloaders=validation_dataloaders,
)

# After training, we can plot the results
plot_tool = probe_emnist_module.visualiaztion_tool(emnist_train_probes)
plot_tool.all_plots(show=False, save_path="plots")

# Now we can create the classifier
mapd_clf, label_encoder = probe_emnist_module.make_mapd_classifier(emnist_train_probes)

# Now we can surface some examples
probe_predictions = probe_emnist_module.mapd_predict(mapd_clf, label_encoder, n_jobs=16)

# Define the possible labels (for plotting)
letters = digits + "ABCDEFGHIJKLMNOPQRSTUVWXYZabdefchnqrt"
labels = {i: l for i, l in enumerate(letters)}

fig = display_surface_predictions(
    probe_predictions,
    mnist_train,
    probe_suite=["typical", "atypical", "random_outputs", "random_inputs_outputs"],
    labels=labels,
    ordered=True,
)
fig.savefig("all.png", dpi=300)
