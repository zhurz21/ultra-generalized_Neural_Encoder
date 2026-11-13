import torch
import math
from torch import nn
from ..modules import *
from ..builder import MODELS


@MODELS.register_module()
class UNet_hologram(nn.Module):
    '''
    UNet model for hologram encoding.
    '''

    def __init__(self, model_params):
        super(UNet_hologram, self).__init__()

        in_channels = model_params['in_channels']   # number of input channels
        out_channels = model_params['out_channels'] # number of output channels
        depth = model_params['depth']               # depth of the network
        wf = model_params['initial_feature_num']    # number of base filters in the first layer of the network
        norm = model_params['norm']                 # normalization type
        up_mode = model_params['up_mode']           # type of upconvolution
        use_dropout = model_params['use_dropout']   # use dropout or not
        dropout_pro = model_params['dropout_pro']   # dropout probability

        assert up_mode in ('upconv', 'upsample')
        self.down_path = nn.ModuleList()            # down sampling path of the network
        self.up_path = nn.ModuleList()              # up sampling path of the network
        self.out_path = nn.ModuleList()             # output path of the network


        prev_channels = int(in_channels)

        for i in range(depth):
            if i != depth - 1:
                self.down_path.append(UNetConvBlock(prev_channels, [wf * (2 ** i), wf * (2 ** i)], 3, 0, norm))
                prev_channels = int(wf * (2 ** i))
                self.down_path.append(nn.AvgPool2d(2))

            else:
                self.down_path.append(UNetConvBlock(prev_channels, [wf * (2 ** i), wf * (2 ** (i - 1))], 3, 0, norm))
                prev_channels = int(wf * (2 ** (i - 1)))

        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, [wf * (2 ** i), int(wf * (2 ** (i - 1)))], up_mode, 3, 0, norm, use_dropout, dropout_pro))
            prev_channels = int(wf * (2 ** (i - 1)))

        self.out_path.append(nn.Conv2d(prev_channels, out_channels, kernel_size=1, padding=0, bias=False))
        

    def forward(self, scat_pot):
        blocks = []   # store the output of each down sampling layer
        x = scat_pot

        for i, down in enumerate(self.down_path):
            x = down(x)
            if i % 2 == 0 and i != (len(self.down_path) - 1):
                blocks.append(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])      # connect the output of the down sampling layer to the up sampling layer
        
        for i, out in enumerate(self.out_path):
            x = out(x)

        return x


