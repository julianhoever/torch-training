from copy import deepcopy
from typing import Any

import torch


class ModelCheckpointHandler:
    def __init__(
        self,
        model: torch.nn.Module,
        initial_metric_value: float = 0,
        increasing_metric: bool = True,
    ) -> None:
        self._checkpoint = self._get_checkpoint(model)
        self._metric_value = initial_metric_value
        self._increasing_metric = increasing_metric

    @staticmethod
    def _get_checkpoint(model: torch.nn.Module) -> dict[str, Any]:
        return deepcopy(model.state_dict())

    def update(self, model: torch.nn.Module, metric_value: float) -> None:
        if (self._increasing_metric and self._metric_value < metric_value) or (
            not self._increasing_metric and self._metric_value > metric_value
        ):
            self._checkpoint = self._get_checkpoint(model)
            self._metric_value = metric_value

    def load_best_checkpoint(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self._checkpoint)
