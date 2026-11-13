import torch
import math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from ..builder import MODELS


@MODELS.register_module()
class Direct_hologram(nn.Module):
    '''
    This model returns the hologram directly as the output without any optimization or encoding.
    '''

    def __init__(self, model_params):
        super(Direct_hologram, self).__init__()

        self.slm_res = model_params['slm_res']
        self.mul_num = model_params['mul_num']

        self.hologram = nn.Parameter(torch.ones((1, 2*self.mul_num, *self.slm_res)))

    def forward(self, input):

        return self.hologram