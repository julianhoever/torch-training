from torch.optim.optimizer import Optimizer


class NoneLRScheduler:
    def __init__(self, optimizer: Optimizer) -> None:
        pass

    def step(self) -> None:
        pass
