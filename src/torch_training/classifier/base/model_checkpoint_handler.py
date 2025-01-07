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
        self._best_checkpoint = self._get_checkpoint(model)
        self._best_metric_value = initial_metric_value
        self._increasing_metric = increasing_metric

    @property
    def best_checkpoint(self) -> dict[str, Any]:
        return self._best_checkpoint

    @property
    def best_metric_value(self) -> float:
        return self._best_metric_value

    @staticmethod
    def _get_checkpoint(model: torch.nn.Module) -> dict[str, Any]:
        return deepcopy(model.state_dict())

    def _is_update_required(self, metric_value: float) -> bool:
        return (self._increasing_metric and self._best_metric_value < metric_value) or (
            not self._increasing_metric and self._best_metric_value > metric_value
        )

    def update(self, model: torch.nn.Module, metric_value: float) -> None:
        if self._is_update_required(metric_value):
            self._best_checkpoint = self._get_checkpoint(model)
            self._best_metric_value = metric_value

    def load_best_checkpoint(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self._best_checkpoint)
