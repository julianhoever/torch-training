from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from torch_training.history import History
from .training_runner import TrainingRunner
from .protocols import LRSchedulerFactory, ForwardPassFn, CountCorrectClassified


def train_classifier(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    forward_pass_fn: ForwardPassFn,
    count_correct_classified: CountCorrectClassified,
    loss_fn: torch.nn.Module,
    lr_scheduler_factory: Optional[LRSchedulerFactory],
    load_best: bool,
    num_workers: int,
    device: Any,
) -> History:
    runner = TrainingRunner(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        forward_pass_fn=forward_pass_fn,
        count_correct_classified=count_correct_classified,
        loss_fn=loss_fn,
        lr_scheduler_factory=lr_scheduler_factory,
        load_best=load_best,
        num_workers=num_workers,
        device=device,
    )
    return runner.run()
