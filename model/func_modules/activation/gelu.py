#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Tuple
from ...builder import AC_FUNC


@AC_FUNC.register_module()
class GELU(nn.GELU):
    """
    Applies the `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ function
    """

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
