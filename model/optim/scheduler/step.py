import math

from torch.optim.lr_scheduler import StepLR
from ...builder import LR_SCHE


@LR_SCHE.register_module()
class StepLRScheduler(StepLR):
    def __init__(self, cfg, optimizer,
                 last_epoch=-1):
        
        step_size = cfg.step_size
        gamma = cfg.gamma

        StepLR.__init__(
            self,
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch
        )