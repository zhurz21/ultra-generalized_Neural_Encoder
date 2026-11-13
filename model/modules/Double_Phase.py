### The following code is from neural-holography/algorithms.py
### The original file is under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.)
### The original file is available at github.com/computational-imaging/neural-holography/

import torch
import math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from ..builder import MODELS


@MODELS.register_module()
class Double_Phase_hologram(nn.Module):
    def __init__(self, model_params=None):
        super(Double_Phase_hologram, self).__init__()

        """converts amplitude and phase to double phase coding

        slm_res: the resolution of the hologram
        three_pi:  if True, outputs values in a 3pi range, instead of 2pi
        mean_adjust:  if True, centers the phases in the range of interest
        """

        self.three_pi = model_params['three_pi'] if ('three_pi' in model_params) else False
        self.mean_adjust = model_params['mean_adjust'] if ('mean_adjust' in model_params) else False


    def double_phase_amp_phase(self, input):
        """converts amplitude and phase to double phase coding

        amplitudes:  per-pixel amplitudes of the complex field
        phases:  per-pixel phases of the complex field
        """
        # extract the amplitude and phase
        slm_field = torch.complex(input[:, 0:1, ...], input[:, 1:2, ...])
        amplitudes = slm_field.abs()
        phases = slm_field.angle()

        # normalize
        amplitudes = amplitudes / amplitudes.max()

        phases_a = phases - torch.acos(amplitudes)
        phases_b = phases + torch.acos(amplitudes)

        phases_out = phases_a
        phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
        phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

        if self.three_pi:
            max_phase = 3 * math.pi
        else:
            max_phase = 2 * math.pi

        if self.mean_adjust:
            phases_out -= phases_out.mean()

        return (phases_out + max_phase / 2) % max_phase - max_phase / 2


    def forward(self, input):

        return self.double_phase_amp_phase(input)