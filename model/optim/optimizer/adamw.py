#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse
from torch.optim import AdamW

from ...builder import OPTIMIZER


@OPTIMIZER.register_module()
class AdamWOptimizer(AdamW):
    """
    `AdamW <https://arxiv.org/abs/1711.05101>`_ optimizer
    """

    def __init__(self, opts, model_params) -> None:
        beta1 = getattr(opts, "beta1", 0.9)
        beta2 = getattr(opts, "beta2", 0.999)
        lr = getattr(opts, "lr", 1e-3)
        ams_grad = getattr(opts, "amsgrad", False)
        eps = getattr(opts, "eps", 1e-08)
        weight_decay = getattr(opts, "weight_decay", 0.01)
        AdamW.__init__(
            self,
            params=model_params,
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=ams_grad,
        )


    def __repr__(self) -> str:
        group_dict = dict()
        for i, group in enumerate(self.param_groups):
            for key in sorted(group.keys()):
                if key == "params":
                    continue
                if key not in group_dict:
                    group_dict[key] = [group[key]]
                else:
                    group_dict[key].append(group[key])

        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        for k, v in group_dict.items():
            format_string += "\t {0}: {1}\n".format(k, v)
        format_string += ")"
        return format_string
