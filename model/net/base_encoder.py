# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint_fn
from typing import Optional, Dict, Tuple, Union, Any
import argparse

from model.builder import get_norm_tuple
from utils.misc import parameter_list
from utils.init_utils import initialize_weights


class BaseEncoder(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, opts) -> None:
        super().__init__()

        self.action_downscale_mode = getattr(opts, "action_downscale_mode", "f_fourier")
        self.action_upscale_mode = getattr(opts, "action_upscale_mode", "f_fourier")
        self.in_channels = getattr(opts, "in_channels", 10)
        self.out_channels = getattr(opts, "out_channels", 2)
        self.depth = getattr(opts, "depth", 4)
        self.norm  = getattr(opts, "norm", "no")


    def reset_parameters(self, opts):
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules(), norm_tuple=get_norm_tuple())


    def _forward_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        # Larger models with large input image size may not be able to fit into memory.
        # We can use gradient checkpointing to enable training with large models and large inputs
        return (
            gradient_checkpoint_fn(layer, x)
            if self.gradient_checkpointing
            else layer(x)
        )


    def _forward_classifier(self, x: Tensor, *args, **kwargs) -> Tensor:
        # We add another classifier function so that the classifiers
        # that do not adhere to the structure of BaseEncoder can still
        # use neural augmentor
        x = self._extract_features(x)
        x = self.classifier(x)
        return x
    

    def forward(self, x: Any, *args, **kwargs) -> Any:
        if self.neural_augmentor is not None:
            if self.training:
                x_aug = self.neural_augmentor(x)
                prediction = self._forward_classifier(x_aug)  # .detach()
                out_dict = {"augmented_tensor": x_aug, "logits": prediction}
            else:
                out_dict = {
                    "augmented_tensor": None,
                    "logits": self._forward_classifier(x),
                }
            return out_dict
        else:
            x = self._forward_classifier(x, *args, **kwargs)
            return x

    def freeze_norm_layers(self) -> None:
        """Freeze normalization layers"""
        for m in self.modules():
            if isinstance(m, get_norm_tuple()):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """Get trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)
