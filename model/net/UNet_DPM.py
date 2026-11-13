import torch
import math
from torch import nn
from ..modules import *
from ..builder import MODELS
from .UNet import UNet_hologram
import utils as utils
import numpy as np

@MODELS.register_module()
class UNet_DPM_hologram(UNet_hologram, AA_Double_Phase_hologram):
    '''
    UNet model for neural-assisted hologram encoding.
    '''

    def __init__(self, model_params):
        super().__init__(model_params)

        self.out_channels = model_params['out_channels']  # number of output channels

    
    def forward(self, scat_pot):

        blocks = []
        x = scat_pot

        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0 and i != (len(self.down_path) - 1):
                blocks.append(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        
        for i, out in enumerate(self.out_path):
            x = out(x)

        # the two channels of the network's output are regarded as 
        # the real and imaginary parts of the hologram
        output_real = x[:,:self.out_channels//2,:,:]
        output_img = x[:,self.out_channels//2:,:,:]
        output_cat = torch.cat([output_real, output_img], 1) 

        # apply Gaussian convolution if sigma > 0
        # this is the pivotal operation of the AA-DPM
        if self.sigma > 0.0:
            conved_input = self.gaussian_conv(output_cat)
        else:
            conved_input = output_cat

        return self.double_phase_amp_phase(conved_input)


