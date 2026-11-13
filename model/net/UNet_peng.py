### The following code is from neural-holography/holonet.py
### The original file is under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.)
### The original file is available at github.com/computational-imaging/neural-holography/

import torch
import torch.nn as nn
from ..modules import Conv2dSame, UnetSkipConnectionBlock
from ..builder import MODELS


@MODELS.register_module()
class Unet_peng(nn.Module):
    '''A 2d-Unet implementation with sane defaults.
    '''

    def __init__(self,
                 model_params):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param nf0: Number of features at highest level of U-Net
        :param num_down: Number of downsampling stages.
        :param max_channels: Maximum number of channels (channels multiply by 2 with every downsampling stage)
        :param use_dropout: Whether to use dropout or no.
        :param dropout_prob: Dropout probability if use_dropout=True.
        :param upsampling_mode: Which type of upsampling should be used. See "UpBlock" for documentation.
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param outermost_linear: Whether the output layer should be a linear layer or a nonlinear one.
        '''
        super(Unet_peng, self).__init__()

        in_channels = model_params['in_channels']
        out_channels = model_params['out_channels']
        nf0 = model_params['nf0']
        num_down = model_params['num_down']
        max_channels = model_params['max_channels']
        use_dropout = model_params['use_dropout']
        upsampling_mode = model_params['upsampling_mode'] if ('upsampling_mode' in model_params) else 'transpose'
        dropout_prob = model_params['dropout_prob'] if ('dropout_prob' in model_params) else 0.1
        norm = model_params['norm'] if ('norm' in model_params) else nn.BatchNorm2d
        outermost_linear = model_params['outermost_linear'] if ('outermost_linear' in model_params) else False

        assert (num_down > 0), "Need at least one downsampling layer in UNet."

        # Define the in block
        self.in_layer = [Conv2dSame(in_channels, nf0, kernel_size=3, bias=True if norm is None else False)]
        if norm is not None:
            self.in_layer += [norm(nf0, affine=True)]
        self.in_layer += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            self.in_layer += [nn.Dropout2d(dropout_prob)]
        self.in_layer = nn.Sequential(*self.in_layer)

        # Define the center UNet block
        self.unet_block = UnetSkipConnectionBlock(min(2 ** (num_down-1) * nf0, max_channels),
                                                  min(2 ** (num_down-1) * nf0, max_channels),
                                                  use_dropout=use_dropout,
                                                  dropout_prob=dropout_prob,
                                                  norm=None, # Innermost has no norm (spatial dimension 1)
                                                  upsampling_mode=upsampling_mode)

        for i in list(range(0, num_down - 1))[::-1]:
            self.unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, max_channels),
                                                      min(2 ** (i + 1) * nf0, max_channels),
                                                      use_dropout=use_dropout,
                                                      dropout_prob=dropout_prob,
                                                      submodule=self.unet_block,
                                                      norm=norm,
                                                      upsampling_mode=upsampling_mode)

        # Define the out layer. Each unet block concatenates its inputs with its outputs - so the output layer
        # automatically receives the output of the in_layer and the output of the last unet layer.
        self.out_layer = [Conv2dSame(2 * nf0,
                                     out_channels,
                                     kernel_size=3,
                                     bias=outermost_linear or (norm is None))]

        if not outermost_linear:
            if norm is not None:
                self.out_layer += [norm(out_channels, affine=True)]
            self.out_layer += [nn.ReLU(True)]

            if use_dropout:
                self.out_layer += [nn.Dropout2d(dropout_prob)]
        self.out_layer = nn.Sequential(*self.out_layer)

        self.out_layer_weight = self.out_layer[0].weight

    def forward(self, x):
        in_layer = self.in_layer(x)
        unet = self.unet_block(in_layer)
        out_layer = self.out_layer(unet)
        return out_layer