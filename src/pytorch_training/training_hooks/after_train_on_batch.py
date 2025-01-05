from typing import Protocol, runtime_checkable


@runtime_checkable
class AfterTrainOnBatchHook(Protocol):
    def after_train_on_batch(self) -> None: ...
