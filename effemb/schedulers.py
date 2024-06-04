import gin
import torch
import math
from abc import ABC, abstractmethod


@gin.configurable
class AbstractScheduler(ABC):
    @abstractmethod
    def set_lr(self, step: int):
        raise NotImplementedError

    #def save_state(self):
    #    raise NotImplementedError

    #def load_state(self):
    #    raise NotImplementedError


@gin.configurable
class WarmupCosineScheduler(AbstractScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps, learning_rate, max_steps) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.final_lr = 0

    def set_lr(self, step: int):
        if step <= self.warmup_steps:
            # linear warmup
            lr = self.learning_rate * step / self.warmup_steps
        else:
            # return self.learning_rate
            # cosine decay
            progress = (step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            lr = self.final_lr + 0.5 * (
                self.learning_rate - self.final_lr
            ) * (1 + math.cos(math.pi * progress))

        for g in self.optimizer.param_groups:
            g["lr"] = lr

        return lr


@gin.configurable
class WarmupConstantScheduler(AbstractScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps, learning_rate, max_steps) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_steps = max_steps

    def set_lr(self, step: int):
        if step <= self.warmup_steps:
            # linear warmup
            lr = self.learning_rate * step / self.warmup_steps
        else:
            lr = self.learning_rate

        for g in self.optimizer.param_groups:
            g["lr"] = lr

        return lr


@gin.configurable
class ConstantScheduler(AbstractScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer, learning_rate, **unused_kwargs) -> None:
        for k in unused_kwargs.keys():
            print(f"Unused argument to scheduler init {k}")
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def set_lr(self, step: int):
        lr = self.learning_rate
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        return lr
