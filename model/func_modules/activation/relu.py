#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Tuple
from ...builder import AC_FUNC


@AC_FUNC.register_module()
class ReLU(nn.ReLU):
    """
    Applies Rectified Linear Unit function
    """

    def __init__(self, cfg, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
