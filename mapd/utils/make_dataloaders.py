from copy import deepcopy
from typing import List, Optional

from torch.utils.data import DataLoader

from mapd.probes.probe_suite_generator import ProbeSuiteGenerator


def make_dataloaders(validation_dataloaders: List[DataLoader], probe_suite_dataset: ProbeSuiteGenerator, dataloader_kwargs: Optional[dict] = None):
    default_dataloader_options = {
        "batch_size": 512,
        "num_workers": 1,
        "prefetch_factor": 1
    }

    if dataloader_kwargs is not None:
        default_dataloader_options.update(dataloader_kwargs)

    probe_suite_dataset = deepcopy(probe_suite_dataset)
    probe_suite_dataset.only_probes = True
    probe_suite_dataset.dataset = None

    return [DataLoader(probe_suite_dataset, **default_dataloader_options)] + validation_dataloaders