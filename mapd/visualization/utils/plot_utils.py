# imports
import sys

import pandas as pd
import torch
from src.data.loss_dataset import LossDataset


def get_loss_dataset(loss_dataset_path: str) -> pd.DataFrame:
    """Load the loss dataset"""
    dataset = LossDataset(loss_dataset_path)
    dataset.load()
    df = dataset.df
    return df


def load_loss_dataset(loss_dataset_path: str):
    """Load the loss dataset from the given path"""
    dataset = LossDataset(loss_dataset_path)
    dataset.load()

    df = dataset.df

    print(f"Loaded {len(df)} samples from {loss_dataset_path}")
    print("Size in GB: ", sys.getsizeof(df) / 1024 / 1024 / 1024)

    return df


def load_probe_suite(probe_suite_path: str):
    """Load the probe suite from the given path"""
    probe_suite = torch.load(probe_suite_path)
    return probe_suite


def get_indices_from_probe_suite(suite: list) -> list[int]:
    return [idx for _, idx in suite]


def plot_styles():
    line_styles = ["solid", "dashed", "dashdot", "dotted"]

    marker_list = ["o", "*", "X", "P", "p", "D", "v", "^", "h", "1", "2", "3", "4"]

    marker_colors = [
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:orange",
        "tab:red",
        "tab:pink",
        "tab:olive",
        "tab:brown",
        "tab:cyan",
        "tab:gray",
    ]

    plot_titles = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "imagenet": "ImageNet",
        "speechcommands": "SpeechCommands"
    }

    return line_styles, marker_list, marker_colors, plot_titles


def plot_dicts():
    ratios = {
        "Typical": [],
        "Atypical": [],
        "Random outputs": [],
        "Random inputs and outputs": [],
        "Corrupted": [],
        "Typical [Val]": [],
        "Atypical [Val]": [],
        "Random outputs [Val]": [],
        "Random inputs and outputs [Val]": [],
        "Corrupted [Val]": [],
        "Train": [],
    }

    learned = {
        "Typical": set([]),
        "Atypical": set([]),
        "Random outputs": set([]),
        "Random inputs and outputs": set([]),
        "Corrupted": set([]),
        "Typical [Val]": set([]),
        "Atypical [Val]": set([]),
        "Random outputs [Val]": set([]),
        "Random inputs and outputs [Val]": set([]),
        "Corrupted [Val]": set([]),
        "Train": set([]),
    }

    suite_names = [
        "Typical",
        "Atypical",
        "Random outputs",
        "Random inputs and outputs",
        "Corrupted",
        "Typical [Val]",
        "Atypical [Val]",
        "Random outputs [Val]",
        "Random inputs and outputs [Val]",
        "Corrupted [Val]",
    ]

    return ratios, learned, suite_names
