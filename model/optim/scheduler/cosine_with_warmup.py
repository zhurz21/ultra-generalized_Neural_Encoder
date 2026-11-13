import math

from torch.optim.lr_scheduler import LambdaLR
from ...builder import LR_SCHE



class CosineLRLambda:
    def __init__(self, num_warmup_steps, num_training_steps, num_cycles, end_lr):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.end_lr = end_lr

    def __call__(self, current_step):  # the function formerly known as "bar"
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        return max(self.end_lr, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))


@LR_SCHE.register_module()
class CosineWithWarmupScheduler(LambdaLR):
    def __init__(self, cfg, optimizer,
                 num_cycles: float = 0.5,
                 last_epoch=-1, 
                 verbose=False):
        num_warmup_steps = cfg.num_warmup_steps
        num_training_steps = cfg.num_training_steps
        end_lr = cfg.end_lr
        num_cycles = cfg.num_cycles
        lr_lambda = CosineLRLambda(
            num_warmup_steps, num_training_steps, num_cycles, end_lr)
        super().__init__(optimizer, lr_lambda, last_epoch, verbose)