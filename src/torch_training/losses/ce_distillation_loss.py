from abc import ABC, abstractmethod

import torch
import torch.nn.functional as func


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return func.sigmoid(x)


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return func.softmax(x, dim=-1)


class _DistillationLossBase(torch.nn.Module, ABC):
    def __init__(
        self, loss_fn: torch.nn.Module, temperature: float, soft_target_weight: float
    ) -> None:
        super().__init__()

        if not (0 <= soft_target_weight <= 1):
            raise ValueError("`soft_target_weight` must be in the interval [0, 1].")

        self._loss_fn = loss_fn
        self._temp = temperature
        self._w_soft = soft_target_weight
        self._w_hard = 1 - soft_target_weight

    @abstractmethod
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor,
    ) -> torch.Tensor: ...


class CEDistillationLoss(_DistillationLossBase):
    def __init__(self, temperature: float, soft_target_weight: float) -> None:
        super().__init__(
            loss_fn=torch.nn.CrossEntropyLoss(),
            temperature=temperature,
            soft_target_weight=soft_target_weight,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor,
    ) -> torch.Tensor:
        hot_student_logits = student_logits / self._temp
        hot_teacher_pred = _softmax(teacher_logits / self._temp)

        soft_loss = self._loss_fn(hot_student_logits, hot_teacher_pred) * self._temp**2
        hard_loss = self._loss_fn(student_logits, hard_targets)

        return soft_loss * self._w_soft + hard_loss * self._w_hard


class BCEDistillationLoss(_DistillationLossBase):
    def __init__(self, temperature: float, soft_target_weight: float) -> None:
        super().__init__(
            loss_fn=torch.nn.BCELoss(),
            temperature=temperature,
            soft_target_weight=soft_target_weight,
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: torch.Tensor,
    ) -> torch.Tensor:
        hot_student_pred = _sigmoid(student_logits / self._temp)
        hot_teacher_pred = _sigmoid(teacher_logits / self._temp)

        soft_loss = self._loss_fn(hot_student_pred, hot_teacher_pred) * self._temp**2
        hard_loss = self._loss_fn(_sigmoid(student_logits), hard_targets)

        return soft_loss * self._w_soft + hard_loss * self._w_hard
