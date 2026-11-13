### The following code is from neural-holography/holonet.py
### The original file is under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.)
### The original file is available at github.com/computational-imaging/neural-holography/

import torch
import torch.nn as nn
from .conv_block import Conv2dSame
from ..modules import *


class UpBlock(nn.Module):
    '''A 2d-conv upsampling block with a variety of options for upsampling, and following best practices / with
    reasonable defaults. (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 post_conv=True,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d,
                 upsampling_mode='transpose'):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param post_conv: Whether to have another convolutional layer after the upsampling layer.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        :param upsampling_mode: Which upsampling mode:
                transpose: Upsampling with stride-2, kernel size 4 transpose convolutions.
                bilinear: Feature map is upsampled with bilinear upsampling, then a conv layer.
                nearest: Feature map is upsampled with nearest neighbor upsampling, then a conv layer.
                shuffle: Feature map is upsampled with pixel shuffling, then a conv layer.
        '''
        super().__init__()

        net = list()

        if upsampling_mode == 'transpose':
            net += [nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=True if norm is None else False)]
        elif upsampling_mode == 'bilinear':
            net += [nn.UpsamplingBilinear2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'nearest':
            net += [nn.UpsamplingNearest2d(scale_factor=2)]
            net += [
                Conv2dSame(in_channels, out_channels, kernel_size=3, bias=True if norm is None else False)]
        elif upsampling_mode == 'shuffle':
            net += [nn.PixelShuffle(upscale_factor=2)]
            net += [
                Conv2dSame(in_channels // 4, out_channels, kernel_size=3,
                           bias=True if norm is None else False)]
        else:
            raise ValueError("Unknown upsampling mode!")

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.ReLU(True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        if post_conv:
            net += [Conv2dSame(out_channels,
                               out_channels,
                               kernel_size=3,
                               bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(out_channels, affine=True)]

            net += [nn.ReLU(True)]

            if use_dropout:
                net += [nn.Dropout2d(0.1, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x, skipped=None):
        if skipped is not None:
            input = torch.cat([skipped, x], dim=1)
        else:
            input = x
        return self.net(input)


class DownBlock(nn.Module):
    '''A 2D-conv downsampling block following best practices / with reasonable defaults
    (LeakyReLU, kernel size multiple of stride)
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 prep_conv=True,
                 middle_channels=None,
                 use_dropout=False,
                 dropout_prob=0.1,
                 norm=nn.BatchNorm2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param prep_conv: Whether to have another convolutional layer before the downsampling layer.
        :param middle_channels: If prep_conv is true, this sets the number of channels between the prep and downsampling
                                convs.
        :param use_dropout: bool. Whether to use dropout or not.
        :param dropout_prob: Float. The dropout probability (if use_dropout is True)
        :param norm: Which norm to use. If None, no norm is used. Default is Batchnorm with affinity.
        '''
        super().__init__()

        if middle_channels is None:
            middle_channels = in_channels

        net = list()

        if prep_conv:
            net += [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels,
                              middle_channels,
                              kernel_size=3,
                              padding=0,
                              stride=1,
                              bias=True if norm is None else False)]

            if norm is not None:
                net += [norm(middle_channels, affine=True)]

            net += [nn.LeakyReLU(0.2, True)]

            if use_dropout:
                net += [nn.Dropout2d(dropout_prob, False)]

        net += [nn.ReflectionPad2d(1),
                nn.Conv2d(middle_channels,
                          out_channels,
                          kernel_size=4,
                          padding=0,
                          stride=2,
                          bias=True if norm is None else False)]

        if norm is not None:
            net += [norm(out_channels, affine=True)]

        net += [nn.LeakyReLU(0.2, True)]

        if use_dropout:
            net += [nn.Dropout2d(dropout_prob, False)]

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)




class UnetSkipConnectionBlock(nn.Module):
    '''Helper class for building a 2D unet.
    '''

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 upsampling_mode,
                 norm=nn.BatchNorm2d,
                 submodule=None,
                 use_dropout=False,
                 dropout_prob=0.1):
        super().__init__()

        if submodule is None:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     UpBlock(inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]
        else:
            model = [DownBlock(outer_nc, inner_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm),
                     submodule,
                     UpBlock(2 * inner_nc, outer_nc, use_dropout=use_dropout, dropout_prob=dropout_prob, norm=norm,
                             upsampling_mode=upsampling_mode)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        forward_passed = self.model(x)
        return torch.cat([x, forward_passed], 1)
