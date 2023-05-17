from torch.utils.data import Dataset

from mapd.probes.probe_suite_generator import ProbeSuiteGenerator
from mapd.proxies.proxy_calculator import ProxyCalculator


def make_probe_suites(
    dataset: Dataset,
    label_count: int,
    proxy_calculator: ProxyCalculator,
    num_probes: int = 500
):
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