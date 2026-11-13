import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
global bordersize
import matplotlib.pyplot as plt
import cv2
import torchvision.models
bordersize = 50
from ..builder import LOSS


# calculate the 1D Gaussian distribution vector
def gaussian(window_size, sigma):
    '''
    window_size: int, the size of the Gaussian distribution vector
    sigma: float, the standard deviation of the Gaussian distribution
    '''

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# create a Gaussian kernel by multiplying two one-dimensional Gaussian distribution vectors
# set the channel parameter to expand to 3 channels
def create_window(window_size, channel=1):
    '''
    window_size: int, the size of the Gaussian distribution vector
    channel: int, the number of channels of the Gaussian kernel
    '''

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).double().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, mean_dim=[-2, -1]):
    '''
    img1: Tensor, the first image
    img2: Tensor, the second image
    window_size: int, the size of the Gaussian distribution vector
    window: Tensor, the Gaussian kernel
    size_average: bool, whether to average the SSIM value
    full: bool, whether to return the full value of SSIM
    val_range: float, the value range of the image
    mean_dim: list, the dimensions to average
    '''
    # import ipdb
    # ipdb.set_trace()
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)
    window = window.to(img1.device)
    # import ipdb
    # ipdb.set_trace()
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq # var(x) = Var(X)=E[X^2]-E[X]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = torch.mean(ssim_map, dim=mean_dim)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    '''
    img1: Tensor, the first image
    img2: Tensor, the second image
    window_size: int, the size of the Gaussian distribution vector
    size_average: bool, whether to average the SSIM value
    val_range: float, the value range of the image
    normalize: bool, whether to normalize the SSIM value
    '''

    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
@LOSS.register_module()
class SSIM(torch.nn.Module):
    '''
    The class 'SSIM' provides the Structural Similarity Index (SSIM) loss function.
    '''

    def __init__(self, window_size=7, size_average=True, val_range=None, dim=[-2, -1]):
        super(SSIM, self).__init__()
        '''
        window_size: int, the size of the Gaussian distribution vector
        size_average: bool, whether to average the SSIM value
        val_range: float, the value range of the image
        dim: list, the dimensions to average
        '''

        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.dim = dim

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        '''
        img1: Tensor, the first image
        img2: Tensor, the second image
        '''

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device).type(img1.dtype)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average, mean_dim=self.dim)


@LOSS.register_module()
class MSSSIM(torch.nn.Module):
    '''
    The class 'MSSSIM' provides the Multi-Scale Structural Similarity Index (MS-SSIM) loss function.
    '''

    def __init__(self, window_size=3, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        '''
        window_size: int, the size of the Gaussian distribution vector
        size_average: bool, whether to average the SSIM value
        channel: int, the number of channels of the image
        '''

        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):

        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        device = torch.device('cuda')  # The gpu you are using
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)



@LOSS.register_module()
class Rmse_loss(nn.Module):
    '''
    The class 'Rmse_loss' provides the Root Mean Square Error (RMSE) loss function.
    '''

    def __init__(self):
        super(Rmse_loss, self).__init__()

    def forward(self,out_amp, target_amp):
        '''
        out_amp: Tensor, the amplitude under test
        target_amp: Tensor, the ground truth amplitude
        '''

        mseloss = nn.MSELoss()
        mse = mseloss(out_amp, target_amp)
        rmse = torch.sqrt(mse)

        return rmse


@LOSS.register_module()
class L1_loss(nn.Module):
    '''
    The class 'L1_loss' provides the L1 loss function.
    '''

    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self,out_amp, target_amp):
        '''
        out_amp: Tensor, the amplitude under test
        target_amp: Tensor, the ground truth amplitude
        '''

        L1loss = nn.L1Loss()
        L1 = L1loss(out_amp, target_amp)
        return L1


@LOSS.register_module()
class mse_loss(nn.Module):
    '''
    The class 'mse_loss' provides the Mean Square Error (MSE) loss function.
    '''

    def __init__(self):
        super(mse_loss, self).__init__()

    def forward(self,out_amp, target_amp):
        '''
        out_amp: Tensor, the amplitude under test
        target_amp: Tensor, the ground truth amplitude
        '''

        mseloss = nn.MSELoss()
        mse = mseloss(out_amp, target_amp)
        return mse


@LOSS.register_module()
class phase_gradient(nn.Module):

    """
    The class 'phase_gradient' provides a regularization function to measure the variation(Gradient or Laplace) of the phase of the complex amplitude. 

    This implements a convolution of the phase with a kernel.

    The kernel is a simple 3 by 3 Laplacian kernel here, but you can also try other edge detection methods.
    """
    

    def __init__(self, kernel = None, loss = nn.MSELoss(), device = torch.device('cuda')):
        """
        Parameters
        ----------
        kernel                  : torch.tensor
                                    Convolution filter kernel, 3 by 3 Laplacian kernel by default.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(phase_gradient, self).__init__()
        self.device = device
        self.loss = loss
        if kernel == None:
            self.kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)/8
        else:
            self.kernel = kernel
        self.kernel = Variable(self.kernel.to(self.device))
        

    def forward(self, phase):
        """
        Calculates the phase gradient Loss.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """
        edge_detect = self.functional_conv2d(phase)
        loss_value = self.loss(edge_detect, torch.zeros_like(edge_detect))
        return loss_value

    def functional_conv2d(self, phase):
        """
        Calculates the gradient of the phase.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        edge_detect              : torch.tensor
                                    The computed phase gradient.
        """
        edge_detect = F.conv2d(phase, self.kernel, padding=self.kernel.shape[-1])
        return edge_detect



@LOSS.register_module()
class amplitude_gradient(nn.Module):

    """
    The class 'phase_gradient' provides a regularization function to measure the variation(Gradient or Laplace) of the phase of the complex amplitude. 

    This implements a convolution of the phase with a kernel.

    The kernel is a simple 3 by 3 Laplacian kernel here, but you can also try other edge detection methods.
    """
    

    def __init__(self, kernel = None, loss = nn.MSELoss(), device = torch.device('cuda')):
        """
        Parameters
        ----------
        kernel                  : torch.tensor
                                    Convolution filter kernel, 3 by 3 Laplacian kernel by default.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(amplitude_gradient, self).__init__()
        self.device = device
        self.loss = loss
        if kernel == None:
            self.kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)/8
        else:
            self.kernel = kernel
        self.kernel = Variable(self.kernel.to(self.device))
        

    def forward(self, recon, target, mask):
        """
        Calculates the phase gradient Loss.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """
        edge_detect_recon = self.functional_conv2d(recon) * mask
        edge_detect_target = self.functional_conv2d(target) * mask
        loss_value = self.loss(edge_detect_recon, edge_detect_target)
        return loss_value

    def functional_conv2d(self, phase):
        """
        Calculates the gradient of the phase.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        edge_detect              : torch.tensor
                                    The computed phase gradient.
        """
        edge_detect = F.conv2d(phase, self.kernel, padding=self.kernel.shape[-1]//2)
        return edge_detect


@LOSS.register_module()
class speckle_contrast(nn.Module):

    """
    The class 'speckle_contrast' provides a regularization function to measure the speckle contrast of the intensity of the complex amplitude using C=sigma/mean. Where C is the speckle contrast, mean and sigma are mean and standard deviation of the intensity.

    We refer to the following paper:

    Kim et al.(2020). Light source optimization for partially coherent holographic displays with consideration of speckle contrast, resolution, and depth of field. Scientific Reports. 10. 18832. 10.1038/s41598-020-75947-0. 
    """
    

    def __init__(self, kernel_size = 11, step_size = (1, 1), loss = nn.MSELoss(), device = torch.device('cuda')):
        """
        Parameters
        ----------
        kernel                  : torch.tensor
                                    Convolution filter kernel, 11 by 11 average kernel by default.
        step_size               :tuple
                                    Convolution stride in height and width direction.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(speckle_contrast, self).__init__()
        self.device = device
        self.loss = loss
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size))/(self.kernel_size**2)
        self.kernel = Variable(self.kernel.type(torch.FloatTensor).to(self.device))


    def forward(self, intensity):
        """
        Calculates the speckle contrast Loss.

        Parameters
        ----------
        intensity               : torch.tensor
                                    intensity of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """
        Speckle_C = self.functional_conv2d(intensity)
        loss_value = self.loss(Speckle_C, torch.zeros_like(Speckle_C))
        return loss_value


    def functional_conv2d(self, intensity):
        """
        Calculates the speckle contrast of the intensity.

        Parameters
        ----------
        phase                   : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        Speckle_C               : torch.tensor
                                    The computed speckle contrast.
        """
        mean = F.conv2d(intensity, self.kernel, stride = self.step_size)
        var = torch.sqrt(F.conv2d(torch.pow(intensity, 2), self.kernel, stride = self.step_size) - torch.pow(mean, 2))
        Speckle_C = var/mean
        return Speckle_C

