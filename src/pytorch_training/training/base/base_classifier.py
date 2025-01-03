from typing import Any, Callable, Optional, Protocol
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from src.pytorch_training.model_hooks import AfterTrainOnBatchHook
from src.pytorch_training.history import History
from .model_checkpoint_handler import ModelCheckpointHandler
from .none_lr_scheduler import NoneLRScheduler


LRSchedulerFactory = Callable[[Optimizer], LRScheduler]


class _ForwardPassFn(Protocol):
    def __call__(
        self, model: torch.nn.Module, samples: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _CountCorrectClassified(Protocol):
    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> int: ...


class _BatchConsumerFn(Protocol):
    def __call__(
        self, samples: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _TrainingRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        ds_train: Dataset,
        ds_val: Dataset,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        forward_pass_fn: _ForwardPassFn,
        count_correct_classified: _CountCorrectClassified,
        lr_scheduler_factory: Optional[LRSchedulerFactory],
        load_best: bool,
        num_workers: int,
        device: Any,
    ) -> None:
        self._model = model
        self._ds_train = ds_train
        self._ds_val = ds_val
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._forward_pass_fn = forward_pass_fn
        self._count_correct_classified = count_correct_classified
        self._load_best = load_best
        self._num_workers = num_workers
        self._device = device

        self._optimizer = Adam(self._model.parameters(), lr=learning_rate)
        self._lr_scheduler: NoneLRScheduler | LRScheduler = (
            NoneLRScheduler(self._optimizer)
            if lr_scheduler_factory is None
            else lr_scheduler_factory(self._optimizer)
        )

    def run(self) -> History:
        self._model.to(self._device)

        history = History()
        ckpt_handler = ModelCheckpointHandler(self._model, increasing_metric=True)
        dl_train, dl_val = self._prepare_dataloader()

        for epoch in range(1, self._epochs + 1):
            self._model.train()
            train_loss, train_accuracy = self._for_each_batch(
                dl_train, self._train_on_batch
            )

            self._model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = self._for_each_batch(
                    dl_val, self._eval_on_batch
                )

            if self._load_best:
                ckpt_handler.update(self._model, val_accuracy)

            history.log("epoch", epoch, epoch)
            history.log("loss", train_loss, val_loss)
            history.log("accuracy", train_accuracy, val_accuracy)

            self._print_epoch_info(history)

        if self._load_best:
            ckpt_handler.load_best_checkpoint(self._model)

        return history

    def _prepare_dataloader(self) -> tuple[DataLoader, DataLoader]:
        data_loader = partial(
            DataLoader, batch_size=self._batch_size, num_workers=self._num_workers
        )
        dl_train = data_loader(self._ds_train, shuffle=True)
        dl_val = data_loader(self._ds_val, shuffle=False)
        return dl_train, dl_val

    def _for_each_batch(
        self, dl: DataLoader, batch_consumer_fn: _BatchConsumerFn
    ) -> tuple[float, float]:
        running_loss = 0.0
        correct_predicted = 0
        num_samples = 0

        for samples, labels in dl:
            samples, labels = samples.to(self._device), labels.to(self._device)

            predictions, loss = batch_consumer_fn(samples=samples, labels=labels)

            running_loss += loss.item()
            correct_predicted += self._count_correct_classified(
                predictions=predictions, labels=labels
            )
            num_samples += len(samples)

        final_loss = running_loss / len(dl)
        final_accuracy = correct_predicted / num_samples

        return final_loss, final_accuracy

    def _train_on_batch(
        self, samples: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._model.zero_grad()

        predictions, loss = self._forward_pass_fn(
            model=self._model, samples=samples, labels=labels
        )

        loss.backward()
        self._optimizer.step()
        self._lr_scheduler.step()

        if isinstance(self._model, AfterTrainOnBatchHook):
            self._model.after_train_on_batch()

        return predictions, loss

    def _eval_on_batch(
        self, samples: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_pass_fn(model=self._model, samples=samples, labels=labels)

    def _print_epoch_info(self, history: History) -> None:
        print(
            f"[epoch {history.training['epoch'][-1]}/{self._epochs}] "
            f"train_loss: {history.training['loss'][-1]:.04f} ; "
            f"train_accuracy: {history.training['accuracy'][-1]:.04f} ; "
            f"val_loss: {history.validation['loss'][-1]:.04f} ; "
            f"val_accuracy: {history.validation['accuracy'][-1]:.04f}"
        )


def train_classifier_base(
    model: torch.nn.Module,
    ds_train: Dataset,
    ds_val: Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    forward_pass_fn: _ForwardPassFn,
    count_correct_classified: _CountCorrectClassified,
    lr_scheduler_factory: Optional[LRSchedulerFactory],
    load_best: bool,
    num_workers: int,
    device: Any,
) -> History:
    runner = _TrainingRunner(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        forward_pass_fn=forward_pass_fn,
        count_correct_classified=count_correct_classified,
        lr_scheduler_factory=lr_scheduler_factory,
        load_best=load_best,
        num_workers=num_workers,
        device=device,
    )
    return runner.run()
