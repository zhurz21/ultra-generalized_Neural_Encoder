import collections.abc
import functools
import itertools
import subprocess
import warnings
from collections import abc
from typing import Optional


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs
):
    module_name = kwargs.get("module_name", "")
    with_decay = []
    without_decay = []
    with_decay_param_names = []
    without_decay_param_names = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                    without_decay_param_names.append(module_name + p_name)
                elif param.requires_grad:
                    with_decay.append(param)
                    with_decay_param_names.append(module_name + p_name)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
                without_decay_param_names.append(module_name + p_name)
            elif param.requires_grad:
                with_decay.append(param)
                with_decay_param_names.append(module_name + p_name)
    param_list = [
        {
            "params": with_decay,
            "weight_decay": weight_decay,
            "param_names": with_decay_param_names,
        }
    ]
    if len(without_decay) > 0:
        param_list.append(
            {
                "params": without_decay,
                "weight_decay": 0.0,
                "param_names": without_decay_param_names,
            }
        )
    return param_list