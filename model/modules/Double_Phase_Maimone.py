### The following code is from tensor_holography/main_v1.py
### The original file is licensed under a custom license provided by the MIT Technology Licensing Office.
### The original file is available at github.com/liangs111/tensor_holography
### The original papre is Shi et al., Towards Real-time Photorealistic 3D Holography with Deep Neural Networks (TensorHolo V1), Nature, 2021.



import torch
import math
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from ..builder import MODELS


@MODELS.register_module()
class Double_Phase_Maimone_hologram(nn.Module):
    def __init__(self, model_params):
        super(Double_Phase_Maimone_hologram, self).__init__()

        """converts amplitude and phase to double phase coding

        three_pi:  if True, outputs values in a 3pi range, instead of 2pi
        mean_adjust:  if True, centers the phases in the range of interest
        axis:  which axis to be reduced
        """

        self.clamp = model_params['clamp']
        self.axis = model_params['axis']
        self.amp_max = model_params['amp_max']
        self.phs_max = model_params['phs_max']
        self.phs_max = torch.from_numpy(self.phs_max[np.newaxis, :, np.newaxis, np.newaxis])
        self.adaptive_phs_shift = model_params['adaptive_phs_shift']
        self.normalize = model_params['normalize']


    def double_phase_amp_phase(self, input):
        """converts amplitude and phase to double phase coding

        amplitudes:  per-pixel amplitudes of the complex field
        phases:  per-pixel phases of the complex field
        """
        # extract the amplitude and phase
        slm_field = torch.complex(input[:, 0:1, ...], input[:, 1:2, ...])
        amp = slm_field.abs()
        phs = slm_field.angle()
        self.phs_max = self.phs_max.to(slm_field.device)

        # normalize amplitude
        if self.amp_max is None:
            # avoid acos producing nan
            amp_max = torch.max(amp) + 1e-6
        amp = amp / amp_max

        # clamp maximum value to 1.0
        if self.clamp:
            amp[amp > 1.0-1e-6] = 1.0-1e-6

        # # discard every other pixel
        # if self.axis == 3:    # reduce columns
        #     amp = amp[:,:,:,0::2]
        #     phs_zero_mean = phs_zero_mean[:,:,:,0::2]
        # elif self.axis == 2:  # reduce rows
        #     amp = amp[:,:,0::2,:]
        #     phs_zero_mean = phs_zero_mean[:,:,0::2,:]

        # # compute two phase maps
        # phs_offset = torch.acos(amp)

        # phs_low = phs_zero_mean - phs_offset
        # phs_high = phs_zero_mean + phs_offset 

        # # arrange in checkerboard pattern
        # if self.axis == 3:
        #     phs_1_1 = phs_low[:,:,0::2,:]
        #     phs_1_2 = phs_high[:,:,0::2,:]
        #     phs_2_1 = phs_high[:,:,1::2,:]
        #     phs_2_2 = phs_low[:,:,1::2,:]
        #     phs_only = torch.concat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
        # elif self.axis == 2:
        #     phs_1_1 = phs_low[:,:,:,0::2]
        #     phs_1_2 = phs_high[:,:,:,1::2]
        #     phs_2_1 = phs_high[:,:,:,0::2]
        #     phs_2_2 = phs_low[:,:,:,1::2]
        #     phs_only = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
        # else:
        #     raise ValueError("axis has to be 2 or 3")
        # phs_only = torch.nn.functional.pixel_shuffle(phs_only, 2)
            
        phases_a = phs - torch.acos(amp)
        phases_b = phs + torch.acos(amp)

        phs_only = phases_a
        phs_only[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
        phs_only[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

        # center phase for each color channel
        phs_only = phs_only - torch.mean(phs_only, dim=[2,3], keepdims=True)

        if self.phs_max != None:
            # phs_only = self.tf_wrap_phs(phs_only, phs_max=self.phs_max, adaptive_phs_shift=self.adaptive_phs_shift)
            phs_only = (phs_only + self.phs_max / 2) % self.phs_max - self.phs_max / 2

        if self.normalize:
            phs_max_4d = self.phs_max.repeat(1, phs_only.size()[1], 1, 1)
            phs_only = phs_only / phs_max_4d

        return phs_only


    
    def tf_wrap_phs(self,
                    phs_only,
                    phs_max=[2*np.pi]*3,
                    adaptive_phs_shift=False):
        
        def wrap_less_equal_than_phs_max(phs_only, phs_max, phs_per_channel_max, phs_per_channel_min):
            return phs_only + (phs_max-phs_per_channel_min-phs_per_channel_max) / 2.0
        
        def wrap_greater_than_phs_max(phs_only, phs_max):
            phs_only = phs_only + phs_max/2.0
            phs_only = torch.where(phs_only < 0, phs_only + 2.0*np.pi, phs_only)
            phs_only = torch.where(phs_only > phs_max, phs_only - 2.0*np.pi, phs_only)
            return phs_only

        if phs_max is not None:
            # wrap out-of-range phase
            if adaptive_phs_shift:
                phs_per_channel_list = []
                for i in range(phs_only.size()[1]):
                    phase_per_channel = phs_only[:,i,:,:]
                    phs_max_channel = phs_max[i]
                    phase_per_channel_max = torch.max(phase_per_channel)
                    phase_per_channel_min = torch.min(phase_per_channel)
                    tf_cond = (phase_per_channel_max - phase_per_channel_min) <= phs_max_channel
                    if tf_cond:
                        phase_per_channel = wrap_less_equal_than_phs_max(phase_per_channel, phs_max_channel, phase_per_channel_max, phase_per_channel_min)
                    else: 
                        phase_per_channel = wrap_greater_than_phs_max(phase_per_channel, phs_max_channel)
                    phs_per_channel_list.append(phase_per_channel)
                phs_only = torch.stack(phs_per_channel_list, dim=1)
            else:
                phs_max_4d = phs_max.repeat(1, phs_only.size()[1], 1, 1)
                phs_only = wrap_greater_than_phs_max(phs_only, phs_max_4d) 
        
        return phs_only


    def forward(self, input):

        return self.double_phase_amp_phase(input)
    


@MODELS.register_module()
class AA_Double_Phase_hologram(Double_Phase_Maimone_hologram):
    def __init__(self, model_params):
        super(AA_Double_Phase_hologram, self).__init__(model_params)

        """converts amplitude and phase to double phase coding

        three_pi:  if True, outputs values in a 3pi range, instead of 2pi
        mean_adjust:  if True, centers the phases in the range of interest
        axis:  which axis to be reduced
        """

        self.kernel_width = model_params['kernel_width']
        self.sigma = model_params['sigma']

        if self.sigma > 0.0:
            self.blur_kernel = torch.from_numpy(self.np_gaussian([self.kernel_width, self.kernel_width], self.sigma)).type(torch.float32)


    def np_gaussian(self, shape=(3,3), sigma=0.5, reshape_4d=True):
        """
        modified from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
        2D Gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        
        if reshape_4d:
            h = h[np.newaxis,np.newaxis, :,:]

        return h


    def gaussian_conv(self, input):

        self.blur_kernel = self.blur_kernel.to(input.device)
        
        cpx_real = input[:, 0:1, ...]
        cpx_imag = input[:, 1:2, ...]
        
        cpx_real = F.conv2d(cpx_real, self.blur_kernel, padding='same')
        cpx_imag = F.conv2d(cpx_imag, self.blur_kernel, padding='same')

        return torch.cat([cpx_real, cpx_imag], 1)


    def forward(self, input):
        
        if self.sigma > 0.0:
            conved_input = self.gaussian_conv(input)
        else:
            conved_input = input

        return self.double_phase_amp_phase(conved_input)