from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax

from pytorch_training.history import History
from .base import LRSchedulerFactory, train_classifier


def train_model(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    load_best: bool = True,
    num_workers: int = 1,
    device: Any = "cpu",
) -> History:
    return train_classifier(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        forward_pass_fn=_forward_pass_fn,
        count_correct_classified=_count_correct_classified,
        loss_fn=loss_fn,
        lr_scheduler_factory=lr_scheduler_factory,
        load_best=load_best,
        num_workers=num_workers,
        device=device,
    )


def _forward_pass_fn(
    model: torch.nn.Module,
    samples: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(samples)
    loss = loss_fn(logits, labels)
    predictions = softmax(logits, dim=-1)
    return predictions, loss


def _count_correct_classified(predictions: torch.Tensor, labels: torch.Tensor) -> int:
    predicted = torch.argmax(predictions, dim=-1)
    target = torch.argmax(labels, dim=-1)
    return int((predicted == target).sum().item())
