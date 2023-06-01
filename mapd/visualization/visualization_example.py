# import plot functions
from mapd.visualization import (
    consistently_learned_plot,
    first_learned_plot,
    loss_curve_plot,
    probe_accuracy_plot,
)

class mapd_visualizer():
    def __init__(
            self,
            name: str,
            probe_suite_path: str,
            loss_dataset_path: str,
            output_path: str,
            rio: bool = False
    ) -> None:
        self.name = name
        self.probe_suite_path = probe_suite_path
        self.loss_dataset_path = loss_dataset_path
        self.output_path = output_path
        self.rio = rio

    def consistently_learned_plot(self) -> None:
        consistently_learned_plot(
            name=self.name,
            probe_suite_path=self.probe_suite_path,
            loss_dataset_path=self.loss_dataset_path,
            output_path=self.output_path,
        )

    def first_learned_plot(self) -> None:
        first_learned_plot(
            name=self.name,
            probe_suite_path=self.probe_suite_path,
            loss_dataset_path=self.loss_dataset_path,
            output_path=self.output_path,
        )

    def loss_curve_plot(self) -> None:
        loss_curve_plot(
            name=self.name,
            probe_suite_path=self.probe_suite_path,
            loss_dataset_path=self.loss_dataset_path,
            output_path=self.output_path,
            rio = self.rio,
        )


    def probe_accuracy_plot(self) -> None:
        probe_accuracy_plot(
            name=self.name,
            probe_suite_path=self.probe_suite_path,
            loss_dataset_path=self.loss_dataset_path,
            output_path=self.output_path,
        )
        

    def generate_plots(self) -> None:
        self.consistently_learned_plot()
        self.first_learned_plot()
        self.loss_curve_plot()
        self.probe_accuracy_plot()