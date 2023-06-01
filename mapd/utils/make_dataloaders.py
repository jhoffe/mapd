from copy import copy
from typing import List, Optional

from torch.utils.data import DataLoader

from mapd.probes.probe_suite_generator import ProbeSuiteDataset


def make_dataloaders(
    validation_dataloaders: List[DataLoader],
    probe_suite_dataset: ProbeSuiteDataset,
    dataloader_kwargs: Optional[dict] = None,
):
    """
    Makes dataloaders for the MAPDModule. The first dataloader is the
    probe suite dataset, and the rest are the validation dataloaders.

    Args:
        validation_dataloaders(List[DataLoader]): List of validation dataloaders.
        probe_suite_dataset(ProbeSuiteDataset): The probe suite dataset.
        dataloader_kwargs(Dict[str, Any]): Keyword arguments for the dataloader.

    Returns: List of dataloaders. The first dataloader is the probe suite
    dataset, and the rest are the validation dataloaders.
    """
    default_dataloader_options = {
        "batch_size": 512,
        "num_workers": 1,
        "prefetch_factor": 1,
    }

    if dataloader_kwargs is not None:
        default_dataloader_options.update(dataloader_kwargs)

    probe_suite_dataset = copy(probe_suite_dataset)
    probe_suite_dataset.only_probes = True

    return [
        DataLoader(probe_suite_dataset, **default_dataloader_options)
    ] + validation_dataloaders
