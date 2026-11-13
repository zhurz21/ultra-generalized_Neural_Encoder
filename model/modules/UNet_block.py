import torch
import math
from torch import nn


class LinearBlock(nn.Module):
    '''
    Linear network block.
    '''

    def __init__(self, slm_res, downscale_factor, in_size, out_size, norm, use_dropout=False, dropout_pro=0.1):
        super(LinearBlock, self).__init__()

        self.slm_res = slm_res                # resolution of the SLM
        self.downscale_factor = downscale_factor         # downscale factor for pixel shuffle
        in_features = self.slm_res[0] * self.slm_res[1] * in_size // (self.downscale_factor**2)   # number of input features
        out_features = self.slm_res[0] * self.slm_res[1] * out_size // (self.downscale_factor**2) # number of output features

        self.in_size = in_size
        self.out_size = out_size

        block = []
        # bulid the linear block based on different normalization types
        if norm == 'weight':
            block.append(nn.utils.weight_norm(nn.Linear(in_features, out_features), name='weight'))
            block.append(nn.ReLU())
            if use_dropout:
                block.append(nn.Dropout(p=dropout_pro))

        elif norm == 'batch':
            block.append(nn.Linear(in_features, out_features))
            block.append(nn.ReLU())
            if use_dropout:
                block.append(nn.Dropout(p=dropout_pro))
            block.append(nn.BatchNorm1d(out_features))

        elif norm == 'no':
            block.append(nn.Linear(in_features, out_features))
            block.append(nn.ReLU())
            if use_dropout:
                block.append(nn.Dropout(p=dropout_pro))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = torch.reshape(x, (x.size()[0], -1))
        out = self.block(x)
        out = torch.reshape(out, (out.size()[0], self.out_size, self.slm_res[0]//self.downscale_factor, self.slm_res[1]//self.downscale_factor))
        return out


class UNetConvBlock(nn.Module):
    '''
    Convolutional network block.
    '''

    def __init__(self, in_size, out_size, kersize, padding, norm, use_dropout=False, dropout_pro=0.1, groups=1):
        super(UNetConvBlock, self).__init__()
        '''
        in_size: number of input channels, scalar
        out_size: number of output channels, list[2]
        kersize: kernel size, scalar
        padding: padding size, scalar
        norm: normalization type, can be 'weight', 'batch', 'dropout', 'no'
        use_dropout: use dropout or not
        dropout_pro: dropout probability
        groups: number of groups for group convolution
        '''

        block = []
        if use_dropout:
            norm = 'dropout'

        # bulid the convolutional block based on different normalization types
        if norm == 'weight':
            block.append(nn.ReplicationPad2d(1))
            block.append(nn.utils.weight_norm((nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                                         padding=int(0), bias=True, groups=groups)), name='weight'))
            block.append(nn.ReLU())

            block.append(nn.ReplicationPad2d(1))
            block.append(nn.utils.weight_norm((nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                                         padding=int(0), bias=True, groups=groups)), name='weight'))
            block.append(nn.ReLU())
            

        elif norm == 'batch':
            block.append(nn.ReflectionPad2d(1))
            block.append(nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                   padding=int(padding), bias=True, groups=groups))
            block.append(nn.BatchNorm2d(out_size[0]))
            block.append(nn.ReLU())
            

            block.append(nn.ReflectionPad2d(1))
            block.append(nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                   padding=int(padding), bias=True, groups=groups))
            block.append(nn.BatchNorm2d(out_size[1]))
            block.append(nn.ReLU())
           

        elif norm == 'dropout':
            block.append(nn.ReplicationPad2d(1))
            block.append((nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                    padding=int(0), bias=True, groups=groups)))
            block.append(nn.ReLU())
            block.append(nn.Dropout(dropout_pro))

            block.append(nn.ReplicationPad2d(1))
            block.append((nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                    padding=int(0), bias=True, groups=groups)))
            block.append(nn.ReLU())
            block.append(nn.Dropout(dropout_pro))


        elif norm == 'no':
            block.append(nn.ReplicationPad2d(1))
            block.append((nn.Conv2d(in_size, out_size[0], kernel_size=int(kersize),
                                    padding=int(0), bias=True, groups=groups)))
            block.append(nn.ReLU())

            block.append(nn.ReplicationPad2d(1))
            block.append((nn.Conv2d(out_size[0], out_size[1], kernel_size=int(kersize),
                                    padding=int(0), bias=True, groups=groups)))
            block.append(nn.ReLU())


        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    '''
    Up sampling network block.
    '''
    
    def __init__(self, in_size, out_size, up_mode, kersize, padding, norm, use_dropout=False, dropout_pro=0.1):
        super(UNetUpBlock, self).__init__()
        '''
        in_size: number of input channels, scalar
        out_size: number of output channels, list[2]
        up_mode: up sampling mode, can be 'upconv', 'upsample'
        kersize: kernel size, scalar
        padding: padding size, scalar
        norm: normalization type, can be 'weight', 'batch', 'dropout', 'no'
        use_dropout: use dropout or not
        dropout_pro: dropout probability
        '''

        block = []
        if up_mode == 'upconv':
            block.append(nn.ConvTranspose2d(in_size, in_size,
                         kernel_size=3, stride=2, bias=False, padding=1, output_padding=1))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(mode='bilinear', scale_factor=2))
            block.append(nn.Conv2d(in_size, in_size,
                         kernel_size=1, bias=False))

        self.block = nn.Sequential(*block)
        self.conv_block = UNetConvBlock(
            in_size * 2, out_size, kersize, padding, norm, use_dropout, dropout_pro)


    def forward(self, x, bridge):
        up = self.block(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out