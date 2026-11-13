### The following code is from neural-holography/holonet.py
### The original file is under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.)
### The original file is available at github.com/computational-imaging/neural-holography/

import math
import numpy as np
import torch
import torch.nn as nn

import utils as utils
from ..modules import Double_Phase_hologram
from .UNet_peng import Unet_peng
from ..builder import MODELS
from model.builder import build_net




@MODELS.register_module()
class InitialPhaseUnet(nn.Module):
    """computes the initial input phase given a target amplitude"""
    def __init__(self, model_param):
        super(InitialPhaseUnet, self).__init__()

        net = [Unet_peng(model_param),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp):
        out_phase = self.net(amp)
        return out_phase


@MODELS.register_module()
class FinalPhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a naive SLM amplitude and phase"""
    def __init__(self, model_param):
        super(FinalPhaseOnlyUnet, self).__init__()

        net = [Unet_peng(model_param),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, amp_phase):
        out_phase = self.net(amp_phase)
        return out_phase


class PhaseOnlyUnet(nn.Module):
    """computes the final SLM phase given a target amplitude"""
    def __init__(self, model_param):
        super(PhaseOnlyUnet, self).__init__()

        net = [Unet_peng(model_param),
               nn.Hardtanh(-math.pi, math.pi)]

        self.net = nn.Sequential(*net)

    def forward(self, target_amp):
        out_phase = self.net(target_amp)
        return (torch.ones(1), out_phase)



@MODELS.register_module()
class HoloNet(nn.Module):
    
    def __init__(self, model_param):
        super(HoloNet, self).__init__()

        # submodules
        self.double_phase = Double_Phase_hologram(model_param["double_phase"])
        self.initial_phase = InitialPhaseUnet(model_param["initial_phase"]) if ("initial_phase" in model_param) else None
        self.final_phase_only = FinalPhaseOnlyUnet(model_param["final_phase_only"]) if ("final_phase_only" in model_param) else None
        # self.initial_phase = None
        # self.final_phase_only = None


    def forward(self, input, stage):
        
        if stage == 'stage_1':
            # compute some initial phase, convert to real+imag representation
            if self.initial_phase is not None:
                init_phase = self.initial_phase(input)
            else:
                init_phase = torch.zeros_like(input)

            return init_phase
        
        elif stage == 'stage_2':
            slm_field = torch.complex(input[:, 0:1, ...], input[:, 1:2, ...])
            amp = slm_field.abs()
            ang = slm_field.angle()

            if self.final_phase_only is None:
                return self.double_phase(input)
            else:
                slm_amp_phase = torch.cat((amp, ang), -3)

                return self.final_phase_only(slm_amp_phase)
