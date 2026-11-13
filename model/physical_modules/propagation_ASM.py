import math
import torch
import numpy as np
import utils as utils
import torch.fft
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os 
from .base_propagator import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ..builder import PROPAGATOR



@PROPAGATOR.register_module()
class propagation_PointSource(propagation_Base_pointsouce):
    '''
    This class is used to compute the point source propagation of the input field.
    '''

    def __init__(self, **params):
        super(propagation_PointSource, self).__init__(**params)


    def __call__(self, u_in, coor, direction="backward", dtype=torch.float32):

        # u_in : the list of the amplitude of the input point sources, shape: [batch, distacnes]
        # coor : the coordinates of point sources, shape: [batch, distacne, 3] (w, h, z)

        w = coor[:, :, 0]  # [batch, distance]
        h = coor[:, :, 1]  # [batch, distance]
        z = coor[:, :, 2]  # [batch, distance]

        nh = h / self.feature_size[0]
        nw = w / self.feature_size[1]

        # the maximum supportted region of the point source under the Nyquist sampling condition
        lh = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dh))).to(self.device)  # [batch, distance]
        lh_n = torch.ceil(lh/self.dh)  # [batch, distance]
        lw = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dw))).to(self.device)  # [batch, distance]
        lw_n = torch.ceil(lw/self.dw)  # [batch, distance]

        # the distance between the point source and the pixel
        r = torch.sqrt((h[:, :, None, None] - self.H[:, None, :, :])**2 + 
                       (w[:, :, None, None] - self.W[:, None, :, :])**2 + 
                       z[:, :, None, None]**2)     # [batch, distance, H, W]
        
        # Sptial Phase Matching oeration for the point source
        ini_phase = (((10 * cm) % (638 * nm))/(638 * nm) - (z % self.wavelength)/self.wavelength) * 2 * np.pi

        if direction == "backward":
            swave = u_in[:, :, None, None] * 1/r * torch.exp(-1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]
        else:
            swave = u_in[:, :, None, None] * 1/r * torch.exp(1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]

        # the following two masks are used to select the region of diffracted point sources
        # which are used to avoid the aliasing effect
        mask_h = ((self.nH - self.nh/2)[:,None,:,:] >= (nh - lh_n + 1)[:,:,None,None]) * \
                ((self.nH - self.nh/2)[:,None,:,:] <= (nh + lh_n)[:,:,None,None])
        mask_w = ((self.nW - self.nw/2)[:,None,:,:] >= (nw - lw_n + 1)[:,:,None,None]) * \
                 ((self.nW - self.nw/2)[:,None,:,:] <= (nw + lw_n)[:,:,None,None])
       
        swave = swave * mask_h * mask_w    # [batch, distance, H, W]


        return swave
    

@PROPAGATOR.register_module()
class propagation_PointSourceRSI(propagation_Base_pointsouce):
    '''
    This class is used to compute the point source propagation of the input field with the Rayleigh-Sommerfield I method.
    '''

    def __init__(self, **params):
        super(propagation_PointSourceRSI, self).__init__(**params)


    def __call__(self, u_in, coor, direction="backward", dtype=torch.float32):

        # u_in : the list of the amplitude of the input point sources, shape: [batch, distacnes]
        # coor : the coordinates of point sources, shape: [batch, distacne, 3] (w, h, z)

        w = coor[:, :, 0]  # [batch, distance]
        h = coor[:, :, 1]  # [batch, distance]
        z = coor[:, :, 2]  # [batch, distance]

        nh = h / self.feature_size[0]
        nw = w / self.feature_size[1]

        # the maximum supportted region of the point source under the Nyquist sampling condition
        lh = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dh))).to(self.device)  # [batch, distance]
        lh_n = torch.ceil(lh/self.dh)  # [batch, distance]
        lw = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dw))).to(self.device)  # [batch, distance]
        lw_n = torch.ceil(lw/self.dw)  # [batch, distance]

        # the distance between the point source and the pixel
        r = torch.sqrt((h[:, :, None, None] - self.H[:, None, :, :])**2 + 
                       (w[:, :, None, None] - self.W[:, None, :, :])**2 + 
                       z[:, :, None, None]**2)     # [batch, distance, H, W]
        
        # Sptial Phase Matching oeration for the point source
        ini_phase = (((10 * cm) % (638 * nm))/(638 * nm) - (z % self.wavelength)/self.wavelength) * 2 * np.pi + torch.pi

        # the Rayleigh-Sommerfield I method
        if direction == "backward":
            swave = z[:, :, None, None] * \
                    1/(1j * self.wavelength) * \
                    u_in[:, :, None, None] * \
                    1/r**2 * \
                    torch.exp(-1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]
        else:
            swave = z[:, :, None, None] * \
                    1/(1j * self.wavelength) * \
                    u_in[:, :, None, None] * \
                    1/r**2 * \
                    torch.exp(1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]

        # the following two masks are used to select the region of diffracted point sources
        # which are used to avoid the aliasing effect
        mask_h = ((self.nH - self.nh/2)[:,None,:,:] > (nh - lh_n + 1)[:,:,None,None]) * \
                ((self.nH - self.nh/2)[:,None,:,:] < (nh + lh_n)[:,:,None,None])
        mask_w = ((self.nW - self.nw/2)[:,None,:,:] > (nw - lw_n + 1)[:,:,None,None]) * \
                 ((self.nW - self.nw/2)[:,None,:,:] < (nw + lw_n)[:,:,None,None])

        swave = swave * mask_h * mask_w     # [batch, distance, H, W]

        return swave, mask_h * mask_w


@PROPAGATOR.register_module()
class propagation_PointSource_norm(propagation_Base_pointsouce):
    '''
    This class is used to compute the point source propagation with amplitude normalization.
    '''

    def __init__(self, **params):
        super(propagation_PointSource_norm, self).__init__(**params)


    def __call__(self, u_in, coor, direction="backward", dtype=torch.float32):

        # u_in : the list of the amplitude of the input point sources, shape: [batch, distacnes]
        # coor : the coordinates of point sources, shape: [batch, distacne, 3] (w, h, z)

        w = coor[:, :, 0]  # [batch, distance]
        h = coor[:, :, 1]  # [batch, distance]
        z = coor[:, :, 2]  # [batch, distance]

        nh = h / self.feature_size[0]
        nw = w / self.feature_size[1]

        # the maximum supportted region of the point source under the Nyquist sampling condition
        lh = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dh))).to(self.device)  # [batch, distance]
        lh_n = torch.ceil(lh/self.dh)  # [batch, distance]
        lw = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dw))).to(self.device)  # [batch, distance]
        lw_n = torch.ceil(lw/self.dw)  # [batch, distance]

        # the distance between the point source and the pixel
        r = torch.sqrt((h[:, :, None, None] - self.H[:, None, :, :])**2 + 
                       (w[:, :, None, None] - self.W[:, None, :, :])**2 + 
                       z[:, :, None, None]**2)     # [batch, distance, H, W]
        
        # Sptial Phase Matching oeration for the point source
        ini_phase = (((10 * cm) % (638 * nm))/(638 * nm) - (z % self.wavelength)/self.wavelength) * 2 * np.pi

        # the point source propagation
        if direction == "backward":
            swave = u_in[:, :, None, None] * \
                    torch.exp(-1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]
        else:
            swave = u_in[:, :, None, None] * \
                    torch.exp(1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]

        # the following two masks are used to select the region of diffracted point sources
        # which are used to avoid the aliasing effect
        mask_h = ((self.nH - self.nh/2)[:,None,:,:] > (nh - lh_n + 1)[:,:,None,None]) * \
                ((self.nH - self.nh/2)[:,None,:,:] < (nh + lh_n)[:,:,None,None])
        mask_w = ((self.nW - self.nw/2)[:,None,:,:] > (nw - lw_n + 1)[:,:,None,None]) * \
                 ((self.nW - self.nw/2)[:,None,:,:] < (nw + lw_n)[:,:,None,None])

        swave = swave * mask_h * mask_w     # [batch, distance, H, W]

        # the amplitude normalization
        attenuation = torch.sqrt(torch.sum(mask_h * mask_w, dim=(-1, -2), keepdim=False))
        # attenuation = torch.sqrt((2*lh_n) * (2*lw_n))
        swave = swave / attenuation[:, :, None, None]

        return swave



@PROPAGATOR.register_module()
class propagation_PointSourceRSII(propagation_Base_pointsouce):
    '''
    This class is used to compute the point source propagation with the Rayleigh-Sommerfield II method.
    '''

    def __init__(self, **params):
        super(propagation_PointSourceRSII, self).__init__(**params)


    def __call__(self, u_in, coor, direction="backward", dtype=torch.float32):

        # u_in : the list of the amplitude of the input point sources, shape: [batch, distacnes]
        # coor : the coordinates of point sources, shape: [batch, distacne, 3] (w, h, z)

        w = coor[:, :, 0]  # [batch, distance]
        h = coor[:, :, 1]  # [batch, distance]
        z = coor[:, :, 2]  # [batch, distance]

        nh = h / self.feature_size[0]
        nw = w / self.feature_size[1]

        # the maximum supportted region of the point source under the Nyquist sampling condition
        lh = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dh))).to(self.device)  # [batch, distance]
        lh_n = torch.ceil(lh/self.dh)  # [batch, distance]
        lw = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dw))).to(self.device)  # [batch, distance]
        lw_n = torch.ceil(lw/self.dw)  # [batch, distance]

        # the distance between the point source and the pixel
        r = torch.sqrt((h[:, :, None, None] - self.H[:, None, :, :])**2 + 
                       (w[:, :, None, None] - self.W[:, None, :, :])**2 + 
                       z[:, :, None, None]**2)     # [batch, distance, H, W]
        
        # Sptial Phase Matching oeration for the point source
        ini_phase = (((10 * cm) % (638 * nm))/(638 * nm) - (z % self.wavelength)/self.wavelength) * 2 * np.pi

        # the Rayleigh-Sommerfield II method
        if direction == "backward":
            swave = u_in[:, :, None, None] * z[:, :, None, None] * 1/(r**2) * (1/(2 * torch.pi * r) + 1/(1j * self.wavelength)) * \
                    torch.exp(-1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]
        else:
            swave = u_in[:, :, None, None] * z[:, :, None, None] * 1/(r**2) * (1/(2 * torch.pi * r) + 1/(1j * self.wavelength)) * \
                    torch.exp(1j * (self.k * r + ini_phase[:, :, None, None]))     # [batch, distance, H, W]

        # the following two masks are used to select the region of diffracted point sources
        # which are used to avoid the aliasing effect
        mask_h = ((self.nH - self.nh/2)[:,None,:,:] > (nh - lh_n + 1)[:,:,None,None]) * \
                ((self.nH - self.nh/2)[:,None,:,:] < (nh + lh_n)[:,:,None,None])
        mask_w = ((self.nW - self.nw/2)[:,None,:,:] > (nw - lw_n + 1)[:,:,None,None]) * \
                 ((self.nW - self.nw/2)[:,None,:,:] < (nw + lw_n)[:,:,None,None])

        swave = swave * mask_h * mask_w     # [batch, distance, H, W]


        return swave


@PROPAGATOR.register_module()
class propagation_RSII(propagation_Base_pointsouce):

    def __init__(self, **params):
        super(propagation_RSII, self).__init__(**params)


    def __call__(self, u_in, coor, dtype=torch.float32):

        # u_in : the 2D input field, shape: [batch, 1, H, W]
        # coor : the coordinates of point sources, shape: [batch, distacne, 3] (z, h, w)

        w = coor[:, :, 0]  # [batch, distance]
        h = coor[:, :, 1]  # [batch, distance]
        z = coor[:, :, 2]  # [batch, distance]

        nh = h / self.feature_size[0]
        nw = w / self.feature_size[1]

        lh = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dh))).to(self.device)  # [batch, distance]
        lh_n = torch.ceil(lh/self.dh)  # [batch, distance]
        lw = z * torch.tensor(np.tan(np.arcsin(self.wavelength/2/self.dw))).to(self.device)  # [batch, distance]
        lw_n = torch.ceil(lw/self.dw)  # [batch, distance]

        r = torch.sqrt((h[:, :, None, None] - self.H[:, None, :, :])**2 + 
                       (w[:, :, None, None] - self.W[:, None, :, :])**2 + 
                       z[:, :, None, None]**2)     # [batch, distance, H, W]
        
        ray = z[:, :, None, None] * 1/(r**2) * (1/(2 * torch.pi * r) + 1/(1j * self.wavelength)) * torch.exp(1j * self.k * r)     # [batch, distance, H, W]
        
        swave = u_in * ray     # [batch, distance, H, W]

        # mask_h = ((self.nH[:,None,:,:] - nh[:,:,None,None])>(self.nh/2-lh_n[:,:,None,None]+1)) * \
        #          ((self.nH[:,None,:,:] - nh[:,:,None,None])<(self.nh/2+lh_n[:,:,None,None]))
        # mask_w = ((self.nW[:,None,:,:] - nw[:,:,None,None])>(self.nw/2-lw_n[:,:,None,None]+1)) * \
        #          ((self.nW[:,None,:,:] - nw[:,:,None,None])<(self.nw/2+lw_n[:,:,None,None]))

        mask_h = ((self.nH[:,None,:,:] - self.nh/2) > (nh[:,:,None,None] - lh_n[:,:,None,None] + 1)) * \
                ((self.nH[:,None,:,:] - self.nh/2) < (nh[:,:,None,None] + lh_n[:,:,None,None]))
        mask_w = ((self.nW[:,None,:,:] - self.nw/2) > (nw[:,:,None,None] - lw_n[:,:,None,None] + 1)) * \
                 ((self.nW[:,None,:,:] - self.nw/2) < (nw[:,:,None,None] + lw_n[:,:,None,None]))

        swave = swave * mask_h * mask_w     # [batch, distance, H, W]

        swave = torch.sum(swave, dim=(-2, -1), keepdim=False)     # [batch, distance]

        return swave



@PROPAGATOR.register_module()
class propagation_ASM_PST(propagation_ASM):
    '''
    This class is used to compute the Phase Space Tailoring (PST) propagation of the input field.
    For more details, please refer to the following paper:
    Zhu et al., Three-dimensional computer holography with phase space tailoring. PhotoniX 5, 34 (2024).
    '''

    def __init__(self, params):
        super(propagation_ASM_PST, self).__init__(params)

    
    def filter_H(self, H_exp, x, y, z, FX, FY, dx, dy, device, dtype, filter_x, filter_y):

        if torch.is_tensor(z) and (z.size()[-1] > 1):
            # multiply by distance
            H_exp = torch.mul(H_exp, z[:, :, None, None])

            # band-limited ASM - Matsushima et al. (2009)
            fy_max = 1 / torch.sqrt((2 * z * (1 / y)) ** 2 + 1) / self.wavelength
            fx_max = 1 / torch.sqrt((2 * z * (1 / x)) ** 2 + 1) / self.wavelength

            FX = torch.from_numpy(FX).to(device)
            FY = torch.from_numpy(FY).to(device)
            
            H_filter1 = ((torch.abs(FX[None, None, :, :]) < fx_max[:, :, None, None]) & 
                                    (torch.abs(FY[None, None, :, :]) < fy_max[:, :, None, None])).type(dtype)
            
            # the second filter for the PST
            fy_half = 1 / (filter_y * dy)   # define the half size of the filter in the frequency domain
            fx_half = 1 / (filter_x * dx)

            # the second filter can be a rectangle, a circle, or an ellipse, 
            # you can choose one of them according to your experimental setup
            H_filter2_rect = ((torch.abs(FX) < fx_half) & (torch.abs(FY) < fy_half)).type(dtype)
            H_filter2_circle = (torch.sqrt(torch.abs(FX)**2 + torch.abs(FY)**2) < math.sqrt((fy_half**2 + fx_half**2)/2)).type(dtype)
            H_filter2_ellipse = ((torch.abs(FX)**2*(fy_half**2) + torch.abs(FY)**2*(fx_half**2)) < (fy_half**2 * fx_half**2)).type(dtype)
        
            H_filter = H_filter1 * H_filter2_rect[None, None, :, :]
        
        else:
            # multiply by distance
            H_exp = torch.mul(H_exp, z)

            # band-limited ASM - Matsushima et al. (2009)
            fy_max = 1 / np.sqrt((2 * z * (1 / y)) ** 2 + 1) / self.wavelength
            fx_max = 1 / np.sqrt((2 * z * (1 / x)) ** 2 + 1) / self.wavelength
            H_filter1 = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

            fy_half = 1 / (filter_y * dy)
            fx_half = 1 / (filter_x * dx)
            H_filter2_rect = torch.tensor(((np.abs(FX) < fx_half) & (np.abs(FY) < fy_half)).astype(np.uint8), dtype=dtype)
            H_filter2_circle = torch.tensor((np.sqrt(np.abs(FX)**2 + np.abs(FY)**2) < np.sqrt((np.abs(fy_half)**2 + np.abs(fx_half)**2)/2)).astype(np.uint8), dtype=dtype)
            H_filter2_ellipse = torch.tensor(((np.abs(FX)**2*np.abs(fy_half)**2 + np.abs(FY)**2*np.abs(fx_half)**2) < (np.abs(fy_half)**2 * np.abs(fx_half)**2)).astype(np.uint8), dtype=dtype)
        
            H_filter = H_filter1 * H_filter2_rect

        # get real/img components
        H_real, H_imag = utils.polar_to_rect(H_filter.to(device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = utils.ifftshift(H)
        H = torch.view_as_complex(H)

        return H
    

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



@PROPAGATOR.register_module()
class propagation_ASM_dense(propagation_ASM):
    '''
    This class is used to compute the ASM propagation with a denser sampling rate.
    For more details, please refer to the following paper:
    Chen et al., Phase hologram optimization with bandwidth constraint strategy for speckle free optical reconstruction. Opt Express. 2021.
    '''

    def __init__(self, params):
        super(propagation_ASM_dense, self).__init__(params)
    

    def forward(self, u_in, H, linear_conv, input_resolution):

        U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')
        padx = (H.size(2) - U1.size(2))//2   # pad the input field to sampling the field with a denser sampling rate
        pady = (H.size(3) - U1.size(3))//2
        U1 = utils.ifftshift(F.pad(utils.fftshift(U1),pad = (pady,pady,padx,padx), mode="constant", value=0))

        if U1.size()[1] == 1:
            U2 = H * U1.repeat(1, H.size()[1], 1, 1)
        elif U1.size()[1] == H.size()[1]:
            U2 = H * U1
        else:
            raise ValueError("The channel of the input field mast be 1 or H.size()[1]!")

        u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
        # u_out = u_out.resize(origin_size[0], origin_size[1], origin_size[2], origin_size[3])

        if linear_conv:
            # return utils.crop_image(u_out, input_resolution) # using stacked version
            return utils.crop_image(u_out, [i * 2 for i in input_resolution], pytorch=True, stacked_complex=False)  # using complex tensor
        else:
            return u_out
        
    
    def __call__(self, u_in, z, linear_conv=True,
                padtype='zero', return_H=False, precomped_H=None,
                dtype=torch.float32, filter_x=4, filter_y=4):

        if linear_conv:
            u_in, input_resolution = self.pad_4_linear(u_in, padtype)

        if precomped_H is None:
            H = self.comp_H(u_in, dtype, z, filter_x=4, filter_y=4)

            # return for use later as precomputed inputs
            if return_H:
                return H
            
        else:
            # pad the precomputed H to match the increased resolution of the spatial frequency map
            H = torch.unsqueeze(torch.unsqueeze(precomped_H,0),0)
            H = utils.fftshift(H)
            H_real = H.real
            H_imag = H.imag
            H_real = utils.pad_image(H_real, [i * 2 for i in H_real.size()[-2:]], padval=0, stacked_complex=False)
            H_imag = utils.pad_image(H_imag, [i * 2 for i in H_imag.size()[-2:]], padval=0, stacked_complex=False)
            H = torch.complex(H_real, H_imag)
            H = utils.ifftshift(H)
        
        return self.forward(u_in, H, linear_conv, input_resolution)


        
def propagation_mask(mask, dtype=torch.float32):
    '''This function is used to generate the Fourier basis image from the frequency mask.
    Args:
        mask: the frequency mask, shape: [batch, 1, H, W]
        dtype: the data type of the output image
    Returns:
        u_out: the Fourier basis image, shape: [batch, 1, H, W]
    '''

    U1 = torch.ones_like(mask)

    U2 = mask * U1

    u_out = utils.fftshift(torch.fft.ifftn(utils.ifftshift(U2), dim=(-2, -1), norm='ortho'))

    return u_out


@PROPAGATOR.register_module()
class propagation_RSI_encrypted1(propagation_RSI):

    def __init__(self, params):
        super(propagation_RSI_encrypted1, self).__init__(params)

    
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

        # frequency coordinates sampling
        fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * Ly), 1 / (2 * dy) - 0.5 / (2 * Ly), num_y, dtype=torch.float32, device=u_in.device)
        fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * Lx), 1 / (2 * dx) - 0.5 / (2 * Lx), num_x, dtype=torch.float32, device=u_in.device)

        # momentum/reciprocal space
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')

        fy_half = 1 / (filter_y * dy)
        fx_half = 1 / (filter_x * dx)
        H_filter2_rect = ((torch.abs(FX) < fx_half) & (torch.abs(FY) < fy_half)).type(dtype)
        H_filter2_rect = H_filter2_rect.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        H = torch.fft.fftn(utils.ifftshift(h), dim=(-2, -1)) * dx * dy * utils.ifftshift(H_filter2_rect)

        if z >= 0:
            self.prop_flag = 0
        else:
            self.prop_flag = 1

        return H