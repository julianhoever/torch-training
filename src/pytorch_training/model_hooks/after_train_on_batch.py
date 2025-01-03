from abc import ABC, abstractmethod


class AfterTrainOnBatchHook(ABC):
    @abstractmethod
    def after_train_on_batch(self) -> None: ...
