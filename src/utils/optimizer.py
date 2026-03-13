import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Callable
from functools import partial


class AdamWMuonWrapper:
    def __init__(self, adam: Optimizer, muon: Optimizer):
        self.muon = muon
        self.adam = adam

    def zero_grad(self):
        self.muon.zero_grad()
        self.adam.zero_grad()

    def step(self):
        self.muon.step()
        self.adam.step()


class CosineLogLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, k: int, T_max: int, last_epoch: int = -1):
        """
        Implementation of Cosine Log Scheduler from RealMLP
        """
        self.k = k
        self.T_max = T_max
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        This is where the magic happens.
        Calculate and return the new learning rate for each parameter group.

        Available helpful attributes:
        - self.last_epoch: The current epoch or step number.
        - self.base_lrs: A list of the initial learning rates for each param group.
        - self.optimizer: The optimizer being wrapped.
        """
        t = self.last_epoch / self.T_max
        return [
            cos_log_scheduler(t=t, lr=base_lr, k=self.k) for base_lr in self.base_lrs
        ]


def cos_log_scheduler(t: float, lr: float, k: int):
    return lr * (0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + (2**k - 1) * t))))


# Code courtesy of Qwen3.5
class WDScheduler:
    """
    Efficient wrapper to schedule weight decay for an optimizer's parameter groups.

    Args:
        optimizer: The PyTorch optimizer.
        scheduler_func: A callable accepting (step, total_steps) returning the target weight_decay.
                       If None, assumes a constant rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler_func: Optional[Callable[[int, int], float]] = None,
        T_max: Optional[int] = None,
    ):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.T_max = T_max
        self.t = 0

    def step(self):
        """
        Updates weight_decay for all parameter groups.
        """
        if self.scheduler_func is None:
            # If no scheduler provided, ensure constant rate (or no-op if already set)
            return

        # Calculate target rate based on provided function
        target_wd = self.scheduler_func(self.t, self.T_max)

        # Change the weights for these classes
        for group in self.optimizer.param_groups:
            group["weight_decay"] = target_wd

        # Increase the internal step
        self.t += 1


def flat_cos_scheduler(t: int, T_max: int, *args):
    ratio = t / T_max
    return 0.5 * (1 + np.cos(np.pi * (max(1, 2 * ratio) - 1)))


FlatCosineWD = partial(WDScheduler, scheduler_func=flat_cos_scheduler)
