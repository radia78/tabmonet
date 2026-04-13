import numpy as np
from typing import Optional
from torch.optim import Optimizer


class WDScheduler:
    """
    Efficient wrapper to schedule weight decay for an optimizer's parameter groups.

    Args:
        optimizer: The PyTorch optimizer.
        scheduler_func: A callable accepting (step, total_steps) returning the target weight_decay.
                       If None, assumes a constant rate.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.t = 0

    @staticmethod
    def scheduler_func():
        pass

    def step(self):
        """
        Updates weight_decay for all parameter groups.
        """

        # Calculate target rate based on provided function
        factor = self.scheduler_func()

        # Change the weights for these classes
        for group in self.optimizer.param_groups:
            group["weight_decay"] *= factor

        # Increase the internal step
        self.t += 1


class FlatCosineLR(WDScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: Optional[int] = None,
    ):
        super().__init__(optimizer)
        self.T_max = T_max

    @staticmethod
    def flat_cos_scheduler(t: int, T_max: int):
        ratio = t / T_max
        return 0.5 * (1 + np.cos(np.pi * (max(1, 2 * ratio) - 1)))

    def step(self):
        """
        Updates weight_decay for all parameter groups.
        """

        # Calculate target rate based on provided function
        factor = self.flat_cos_scheduler(self.t, self.T_max)

        # Change the weights for these classes
        for group in self.optimizer.param_groups:
            group["weight_decay"] *= factor

        # Increase the internal step
        self.t += 1


class StepLR(WDScheduler):
    def __init__(self, optimizer: Optimizer, factor: float):
        super().__init__(optimizer)
        self.factor = factor

    def step(self):
        """
        Updates weight_decay for all parameter groups.
        """
        # Change the weights for these classes
        for group in self.optimizer.param_groups:
            group["weight_decay"] *= self.factor

        # Increase the internal step
        self.t += 1
