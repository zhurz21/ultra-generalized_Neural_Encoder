import torch
import math
from torch import nn



class ConvBlock(nn.Module):
    '''
    Convolutional block with different normalization and parameter initialization options.
    '''

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 padding, 
                 norm,
                 w_init,
                 w_init_params, 
                 w_init_r, 
                 b_init, 
                 b_init_params,
                 b_init_std,
                 activate_func):
        super(ConvBlock, self).__init__()
        '''
        args:
        in_channels: number of input channels, scalar
        out_channels: number of output channels, scalar
        kernel_size: kernel size, scalar
        padding: padding type, 'same' or 'valid'
        norm: normalization type, 'weight', 'batch' or 'no'
        w_init: weight initialization type, 'xavier' or 'truncated_normal'
        '''
        block = []

        if norm == 'weight':
            conv_block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
            block.append(nn.utils.weight_norm((conv_block), name='weight'))
            block.append(activate_func)


        elif norm == 'batch':
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True))
            block.append(nn.BatchNorm2d(out_channels))
            block.append(activate_func)


        elif norm == 'no':
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True))
            block.append(activate_func)


        self.block = nn.Sequential(*block)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if w_init == 'xavier':
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if b_init == 'truncated_normal':
                    nn.init.trunc_normal_(m.bias, std=b_init_std)  


    def forward(self, x):
        out = self.block(x)
        return out
    



class Conv2dSame(torch.nn.Module):
    '''2D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    '''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        '''
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        '''
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=1)
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)