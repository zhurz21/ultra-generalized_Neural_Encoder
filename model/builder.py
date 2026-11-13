from utils import Registry
from typing import Optional
import torch


MODELS = Registry('model')
LOSS = Registry('loss')
AC_FUNC = Registry('activation function')
OPTIMIZER = Registry('optimizer')
NORM_FUNC = Registry('norm function')
LR_SCHE = Registry('lr scheduler')
PROPAGATOR = Registry('propagator')


def get_norm_tuple():
    module_dict = NORM_FUNC.module_dict
    dict_tuple = list(module_dict.values())
    return tuple(dict_tuple)


def build_net(cfg):
    """Build net."""
    return MODELS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSS.build(cfg)


def build_propagator(cfg):
    """Build propagator."""
    return PROPAGATOR.build(cfg)


def build_ac_func(cfg) -> torch.nn.Module:
    """Build activation function."""

    act_layer = None

    act_layer = AC_FUNC.build(cfg)
    return act_layer


def build_optimizer(cfg, **kwargs):
    """Build optimizer."""
    return OPTIMIZER.build(cfg, **kwargs)


def build_norm_func(cfg,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    ) -> torch.nn.Module:
    """Build norm function."""

    if cfg.use_norm:
        norm_type = (
            getattr(cfg, "name", "batch_norm")
            if norm_type is None
            else norm_type
        )
        num_groups = (
            getattr(cfg, "groups", 1)
            if num_groups is None
            else num_groups
        )
        momentum = getattr(cfg, "momentum", 0.1)
        norm_layer = None
        norm_type = norm_type.lower() if norm_type is not None else None

        if torch.cuda.device_count() < 1 and norm_type.find("sync_batch") > -1:
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_FUNC.build(cfg, 
                                     num_features=num_features,
                                     norm_type=norm_type,
                                     num_groups=num_groups
                                     )
        
        return norm_layer
    

def build_lr_sche(cfg, **kwargs):
    """Build lr scheduler"""
    if cfg.use_lr_scheduler:
        return LR_SCHE.build(cfg, **kwargs)
    else:
        return None
    


def get_activation_fn(opts) -> torch.nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """
    return build_ac_func(cfg=opts)


def get_normalization_layer(
        opts,
        num_features: int,
        norm_type: Optional[str] = None,
        num_groups: Optional[int] = None,) -> torch.nn.Module:
    """
    Helper function to get normalization layer
    """
    return build_norm_func(cfg=opts,
                           num_features=num_features,
                           norm_type=norm_type,
                           num_groups=num_groups)