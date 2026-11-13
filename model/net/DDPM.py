### The following code is from tensor_holography/main_v2.py
### The original file is licensed under a custom license provided by the MIT Technology Licensing Office.
### The original file is available at github.com/liangs111/tensor_holography
### The original papre is Shi et al., End-to-end Learning of 3D Phase-only Holograms for Holographic Display (TensorHolo V2), LSA, 2022

import torch
import math
from torch import nn
import numpy as np
from ..modules import *
from ..builder import MODELS
import utils as utils

@MODELS.register_module()
class DDPM_hologram(nn.Module):
    def __init__(self, model_params):
        super(DDPM_hologram, self).__init__()

        self.model_params = model_params

        self.model_vars = self._build_model_vars()

        self.aadpm = AA_Double_Phase_hologram(model_params)


    def _build_model_vars(self):
        postfix="_ddpm"
        fw   = np.full((self.model_params["num_layers"+postfix]), self.model_params["filter_width"+postfix], dtype=int)
        fnum = np.append(np.full((self.model_params["num_layers"+postfix]-1), self.model_params["num_filters_per_layer"+postfix], dtype=int), 
                         self.model_params["output_dim"+postfix]*(self.model_params["interleave_rate"+postfix]**2))
        model_vars = {}

        self.conv_path = nn.ModuleList()

        for i in range(self.model_params["num_layers"+postfix]):    
            # first layer
            if i==0:  
                in_dim, out_dim = self.model_params["input_dim"+postfix] * (self.model_params["interleave_rate"+postfix]**2), fnum[i]
                activate_func = self.model_params["activation_func"+postfix]
            # last layer
            elif i==self.model_params["num_layers"+postfix]-1:
                in_dim, out_dim = fnum[i-1] + self.model_params["input_dim"+postfix] * (self.model_params["interleave_rate"+postfix]**2),  \
                                  self.model_params["output_dim"+postfix] * (self.model_params["interleave_rate"+postfix]**2)
                activate_func = self.model_params["output_activation_func"+postfix]
            else:
                in_dim, out_dim = fnum[i-1], fnum[i]  
                activate_func = self.model_params["activation_func"+postfix]

            model_vars[i] = {'kernel_size':(fw[i], fw[i]),
                             'in_channels':in_dim, 
                             'out_channels':out_dim,
                             'padding':'same',
                             'w_init':'xavier',
                             'norm':'batch',
                             'w_init_params':(in_dim, out_dim),
                             'w_init_r':self.model_params["weight_var_scale"+postfix],
                             'b_init':'truncated_normal',
                             'b_init_params':out_dim,
                             'b_init_std':self.model_params["bias_stddev"+postfix],
                             'activate_func':activate_func,
                            }
            
            self.conv_path.append(ConvBlock(**model_vars[i]))

        return model_vars


    def forward(self, input):
        
        in_field = torch.complex(input[:, 0:1, ...], input[:, 1:2, ...])

        in_abs = in_field.abs()
        in_angle = in_field.angle() / 2.0 / torch.pi + 0.5

        x_in = torch.cat([in_abs, in_angle], 1)

        postfix="_ddpm"
        layers = {}
        prev_layers = {}

        if self.model_params["renormalize_input"+postfix]:
            x_in = x_in - 0.5

        # build graph
        for i in range(self.model_params["num_layers"+postfix]):
            if i==0:
                prev_layers[i] = x_in   
            elif (i<3) or (i%2==0): 
                prev_layers[i] = layers[i-1]
            else: 
                prev_layers[i] = layers[i-1] + prev_layers[i-2]
            
            if i == self.model_params["num_layers"+postfix]-1:
                prev_layers[i] = torch.cat([prev_layers[i], x_in], dim = 1)

            if not i == self.model_params["num_layers"+postfix]-1:
                layers[i] = self.conv_path[i](prev_layers[i])
            else:
                # last layer
                field = self.conv_path[i](prev_layers[i])

                # normalize amplitude to [0, sqrt(2)]
                # normalize phase to [0, 1]
                # compute complex field, the phase is renormalized to [-pi, pi]
                amp = field[:,:self.model_params["output_dim"+postfix]//2,:,:]*np.sqrt(0.5) + np.sqrt(0.5)
                phs = field[:,self.model_params["output_dim"+postfix]//2:,:,:]*0.5 + 0.5

                slm_real, slm_imag = utils.polar_to_rect(amp, (phs-0.5)*2.0*np.pi)
                out_field = torch.complex(slm_real, slm_imag)

                output_real = out_field.real
                output_img = out_field.imag
                output_cat = torch.cat([output_real, output_img], 1)
                
                slm_phase = self.aadpm(output_cat)

        return slm_phase


