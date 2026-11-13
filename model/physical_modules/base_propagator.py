import math
import torch
import numpy as np
import utils as utils
import torch.fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ..builder import PROPAGATOR
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9




@PROPAGATOR.register_module()
class propagation_Base_pointsouce():
    '''
    This basic point source class.
    The coordinate grids are defined in this class.
    self.H, self.W : the 2D grids of the input field, shape: [1, H, W]
    self.nH, self.nW : the 2D grids of the pixel index, shape: [1, H, W]
    '''

    def __init__(self, feature_size, channel, wavelength, slm_res, device):
        
        self.feature_size = feature_size
        self.channel = channel
        self.wavelength = wavelength
        self.slm_res = slm_res
        self.device = device

        self.k = 2 * torch.pi / self.wavelength

        dh, dw = self.feature_size
        nh, nw = self.slm_res

        self.dh, self.dw = dh, dw
        self.nh, self.nw = nh, nw

        h = torch.arange(start=-(nh//2*dh-dh/2), end=nh//2*dh, step=dh).to(self.device)
        w = torch.arange(start=-(nw//2*dw-dw/2), end=nw//2*dw, step=dw).to(self.device)
        self.H, self.W = torch.meshgrid(h, w, indexing='ij')

        self.W = self.W.unsqueeze(0)
        self.H = self.H.unsqueeze(0)

        nh = torch.arange(start=0, end=nh, step=1).to(self.device)
        nw = torch.arange(start=0, end=nw, step=1).to(self.device)
        self.nH, self.nW = torch.meshgrid(nh, nw, indexing='ij')

        self.nW = self.nW.unsqueeze(0)
        self.nH = self.nH.unsqueeze(0)



@PROPAGATOR.register_module()
class propagation_ASM():
    '''
    This basic ASM class is used to compute the angular spectrum method (ASM) propagation of the input field.
    This class is copied and modified from neural-holography/propagation_ASM.py
    The original file is under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.)
    The original file is available at github.com/computational-imaging/neural-holography/
    '''

    def __init__(self, params):
        
        self.feature_size = params['feature_size']
        self.channel = params['channel']
        self.wavelength = params['wavelength']
        self.slm_res = params['slm_res']

    
    def comp_H(self, u_in, dtype, z, filter_x=4, filter_y=4):

        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = self.feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y, dtype=np.float32)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x, dtype=np.float32)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / self.wavelength ** 2 - (FX ** 2 + FY ** 2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

        H = self.filter_H(H_exp, x, y, z, FX, FY, dx, dy, u_in.device, dtype, filter_x, filter_y)

        return H

    
    def filter_H(self, H_exp, x, y, z, FX, FY, dx, dy, device, dtype, filter_x=4, filter_y=4):

        if torch.is_tensor(z) and (z.size()[-1] > 1):
            # multiply by distance
            H_exp = torch.mul(H_exp, z[:, :, None, None])

            # band-limited ASM - Matsushima et al. (2009)
            fy_max = 1 / torch.sqrt((2 * z * (1 / y)) ** 2 + 1) / self.wavelength
            fx_max = 1 / torch.sqrt((2 * z * (1 / x)) ** 2 + 1) / self.wavelength

            FX = torch.from_numpy(FX).to(device)
            FY = torch.from_numpy(FY).to(device)
            
            H_filter = ((torch.abs(FX[None, None, :, :]) < fx_max[:, :, None, None]) & 
                                    (torch.abs(FY[None, None, :, :]) < fy_max[:, :, None, None])).type(dtype)
        
        else:
            # multiply by distance
            H_exp = torch.mul(H_exp, z)

            # band-limited ASM - Matsushima et al. (2009)
            fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / self.wavelength
            fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / self.wavelength
            H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)


        # get real/img components
        H_real, H_imag = utils.polar_to_rect(H_filter.to(device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = utils.ifftshift(H)
        H = torch.view_as_complex(H)

        return H
    


    def pad_4_linear(self, u_in, padtype):
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

        return u_in, input_resolution
    

    def forward(self, u_in, H, linear_conv, input_resolution):

        U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')

        if U1.size()[1] == 1:
            U2 = H * U1.repeat(1, H.size()[1], 1, 1)
        elif U1.size()[1] == H.size()[1]:
            U2 = H * U1
        else:
            raise ValueError("The channel of the input field mast be 1 or H.size()[1]!")

        u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

        if linear_conv:
            # return utils.crop_image(u_out, input_resolution) # using stacked version
            return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
        else:
            return u_out
    


    def __call__(self, u_in, z, linear_conv=True,
                padtype='zero', return_H=False, precomped_H=None,
                dtype=torch.float32, filter_x=4, filter_y=4):

        if linear_conv:
            u_in, input_resolution = self.pad_4_linear(u_in, padtype)

        if precomped_H is None:
            H = self.comp_H(u_in, dtype, z, filter_x=4, filter_y=4)
        else:
            H = precomped_H

        # return for use later as precomputed inputs
        if return_H:
            return H
        
        return self.forward(u_in, H, linear_conv, input_resolution)


@PROPAGATOR.register_module()
class propagation_RSI():

    def __init__(self, params):
        
        self.feature_size = params['feature_size']
        self.channel = params['channel']
        self.wavelength = params['wavelength']
        self.slm_res = params['slm_res']
        self.k = 2 * math.pi / self.wavelength

    
    def comp_H(self, u_in, dtype, z, filter_x=4, filter_y=4):

        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = self.feature_size

        # size of the field
        Ly, Lx = (dy * float(num_y), dx * float(num_x))

        y = torch.linspace(-Ly/2, Ly/2-dy, num_y, dtype=dtype, device=u_in.device)
        x = torch.linspace(-Lx/2, Lx/2-dx, num_x, dtype=dtype, device=u_in.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        r = torch.sqrt(X**2 + Y**2 + z**2)
        h = 1/(2 * torch.pi) * abs(z) / r * (1/r - 1j * self.k) * torch.exp(1j * self.k * r) / r
        h = h.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # # frequency coordinates sampling
        # fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * Ly), 1 / (2 * dy) - 0.5 / (2 * Ly), num_y, dtype=torch.float32, device=u_in.device)
        # fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * Lx), 1 / (2 * dx) - 0.5 / (2 * Lx), num_x, dtype=torch.float32, device=u_in.device)

        # # momentum/reciprocal space
        # FX, FY = torch.meshgrid(fx, fy, indexing='ij')

        # fy_half = 1 / (filter_y * dy)
        # fx_half = 1 / (filter_x * dx)
        # H_filter2_rect = ((torch.abs(FX) < fx_half) & (torch.abs(FY) < fy_half)).type(dtype)

        H = torch.fft.fftn(utils.ifftshift(h), dim=(-2, -1)) * dx * dy

        if z >= 0:
            self.prop_flag = 0
        else:
            self.prop_flag = 1

        return H

    def pad_4_linear(self, u_in, padtype):
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in ** 2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

        return u_in, input_resolution
    

    def forward(self, u_in, H, linear_conv, input_resolution):

        if self.prop_flag == 1:
            # u_in = torch.conj(u_in)
            H = torch.conj(H)

        U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1))

        if U1.size()[1] == 1:
            U2 = H * U1.repeat(1, H.size()[1], 1, 1)
        elif U1.size()[1] == H.size()[1]:
            U2 = H * U1
        else:
            raise ValueError("The channel of the input field mast be 1 or H.size()[1]!")

        u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1)))

        # if self.prop_flag == 1:
        #     u_out = torch.conj(u_out)

        if linear_conv:
            # return utils.crop_image(u_out, input_resolution) # using stacked version
            return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
        else:
            return u_out

    def __call__(self, u_in, z, linear_conv=True,
                padtype='zero', return_H=False, precomped_H=None,
                dtype=torch.float32, filter_x=4, filter_y=4):

        if linear_conv:
            u_in, input_resolution = self.pad_4_linear(u_in, padtype)

        if precomped_H is None:
            H = self.comp_H(u_in, dtype, z, filter_x=4, filter_y=4)
        else:
            H = precomped_H

        # return for use later as precomputed inputs
        if return_H:
            return H
        
        return self.forward(u_in, H, linear_conv, input_resolution)