from collections.abc import Callable
from typing import Protocol

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


LRSchedulerFactory = Callable[[Optimizer], LRScheduler]


class ForwardPassFn(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        samples: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class CountCorrectClassified(Protocol):
    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> int: ...
