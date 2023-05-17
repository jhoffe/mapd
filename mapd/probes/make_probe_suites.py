from typing import Union

from torch.utils.data import Dataset

from mapd.probes.probe_suite_generator import ProbeSuiteGenerator
from mapd.proxies.proxy_calculator import ProxyCalculator
import os


def make_probe_suites(
        dataset: Dataset,
        label_count: int,
        proxy_calculator: Union[str, os.PathLike, ProxyCalculator],
        num_probes: int = 500
):
    """
    Helper function to generate a probe suite from a dataset and proxy calculator.

    Args:
        dataset(Dataset): Dataset to generate probe suite from.
        label_count(int): Number of labels in dataset.
        proxy_calculator(str, os.PathLike, ProxyCalculator): A path to the stored proxy results or a ProxyCalculator object.
        num_probes: Number of probes to generate.

    Returns:
        ProbeSuiteGenerator: A ProbeSuiteGenerator object.
    """

    if isinstance(proxy_calculator, (str, os.PathLike)):
        proxy_calculator = ProxyCalculator(proxy_calculator, "proxy_metric")

    if proxy_calculator.proxy_df is None:
        proxy_calculator.load()

    probe_suite = ProbeSuiteGenerator(
        dataset,
        label_count,
        num_probes=num_probes,
        proxy_calculator=proxy_calculator,
    )
    probe_suite.generate()

    return probe_suite
