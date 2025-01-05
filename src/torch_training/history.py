import csv
from pathlib import Path


class History:
    def __init__(self) -> None:
        self._metrics: dict[str, dict[str, list[float]]] = dict(
            training=dict(), validation=dict()
        )

    @property
    def training(self) -> dict[str, list[float]]:
        return self._metrics["training"]

    @property
    def validation(self) -> dict[str, list[float]]:
        return self._metrics["validation"]

    def log(self, metric: str, training_value: float, validation_value: float) -> None:
        if metric not in self.training:
            self._metrics["training"][metric] = []
            self._metrics["validation"][metric] = []

        self._metrics["training"][metric].append(training_value)
        self._metrics["validation"][metric].append(validation_value)

    def save_csv(self, destination: Path, csv_delimiter: str = ",") -> None:
        columns = list(self.training.keys())

        with destination.open("w", newline="") as out_file:
            csv_writer = csv.writer(out_file, delimiter=csv_delimiter)
            csv_writer.writerow(["split", *columns])

            for split in ["training", "validation"]:
                value_pairs = zip(*[self._metrics[split][name] for name in columns])
                csv_writer.writerows([split, *values] for values in value_pairs)
