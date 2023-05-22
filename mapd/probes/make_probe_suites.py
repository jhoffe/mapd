import os
from typing import Union

from mapd.probes.probe_suite_generator import ProbeSuiteDataset
from mapd.probes.utils.idx_dataset import IDXDataset
from mapd.proxies.proxy_calculator import ProxyCalculator


def make_probe_suites(
    dataset: IDXDataset,
    label_count: int,
    proxy_calculator: Union[str, os.PathLike, ProxyCalculator],
    num_probes: int = 500,
    add_train_suite: bool = False,
):
    """
    Helper function to generate a probe suite from a dataset and proxy calculator.

    Args:
        dataset(IDXDataset): Dataset to generate probe suite from.
        label_count(int): Number of labels in dataset.
        proxy_calculator(str, os.PathLike, ProxyCalculator): A path to the stored
            proxy results or a ProxyCalculator object.
        num_probes(int): Number of probes to generate.
        add_train_suite(bool): Whether to add the train suite to the probe suite.
    Returns:
        ProbeSuiteDataset: A ProbeSuiteGenerator object.
    """

    if isinstance(proxy_calculator, (str, os.PathLike)):
        proxy_calculator = ProxyCalculator(proxy_calculator, "proxy_metric")

    if proxy_calculator.proxy_df is None:
        proxy_calculator.load()

    probe_suite = ProbeSuiteDataset(
        dataset,
        label_count,
        num_probes=num_probes,
        proxy_calculator=proxy_calculator,
        add_train_suite=add_train_suite,
    )
    probe_suite.generate()

    return probe_suite
