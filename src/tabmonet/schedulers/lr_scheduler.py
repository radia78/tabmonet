import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineLogLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, k: int, T_max: int, last_epoch: int = -1):
        """
        Implementation of Cosine Log Scheduler from RealMLP
        """
        self.k = k
        self.T_max = T_max
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def cos_log_scheduler(t: float, lr: float, k: int):
        return lr * (0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + (2**k - 1) * t))))

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
            self.cos_log_scheduler(t=t, lr=base_lr, k=self.k)
            for base_lr in self.base_lrs
        ]
