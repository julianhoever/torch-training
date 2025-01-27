from typing import Any, Optional
from functools import partial

import torch
from torch.utils.data import Dataset
from torch.nn.functional import softmax

from torch_training.history import History
from torch_training.classifier.base import LRSchedulerFactory, train_classifier
from torch_training.losses.ce_distillation_loss import CEDistillationLoss


def train_model(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    temperature: float,
    soft_target_weight: float,
    ds_train: Dataset,
    ds_val: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    load_best: bool = True,
    num_workers: int = 1,
    device: Any = "cpu",
) -> History:
    teacher_model.to(device)
    teacher_model.eval()

    return train_classifier(
        model=student_model,
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        forward_pass_fn=partial(_forward_pass_fn, teacher_model=teacher_model),
        count_correct_classified=_count_correct_classified,
        loss_fn=CEDistillationLoss(temperature, soft_target_weight),
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
    teacher_model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_logits = teacher_model(samples)
    student_logits = model(samples)
    loss = loss_fn(student_logits, teacher_logits, labels)
    student_predictions = softmax(student_logits, dim=-1)
    return student_predictions, loss


def _count_correct_classified(predictions: torch.Tensor, labels: torch.Tensor) -> int:
    predicted = torch.argmax(predictions, dim=-1)
    target = torch.argmax(labels, dim=-1)
    return int((predicted == target).sum().item())
