import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Pad, InterpolationMode
import numpy as np
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9



def psnr2(target_amp, recon_amp):
    '''This function calculates the PSNR between the target and the reconstructed images
    Args:
        target_amp: torch.tensor, target image
        recon_amp: torch.tensor, reconstructed image
    Returns:
        psnr: float, PSNR value
    '''

    target_amp = target_amp
    recon_amp = recon_amp
    mse = torch.mean((target_amp - recon_amp) ** 2, dim=[-1,-2], keepdim=True)
    mse[mse < torch.tensor(1.0e-10).to(mse.device)] = 100
    PIXEL_MAX = torch.tensor(1.).to(mse.device)
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def rmse2(target_amp, recon_amp):
    '''This function calculates the RMSE between the target and the reconstructed images
    Args:
        target_amp: torch.tensor, target image
        recon_amp: torch.tensor, reconstructed image
    Returns:
        rmse: float, RMSE value
    '''

    target_amp = target_amp
    recon_amp = recon_amp
    mse = torch.mean((target_amp - recon_amp) ** 2, dim=[-1,-2], keepdim=True)
    rmse = torch.sqrt(mse)
    return rmse


def ssim2(target_amp, recon_amp, color = False):
    '''This function calculates the SSIM between the target and the reconstructed images
    Args:
        target_amp: torch.tensor, target image
        recon_amp: torch.tensor, reconstructed image
        color: bool, whether the images are color images
    Returns:
        ssim: float, SSIM value
    '''

    target_amp = target_amp.cpu().detach().numpy()[0, 0, :, :]
    recon_amp = recon_amp.cpu().detach().numpy()[0, 0, :, :]
    ssim = compare_ssim(target_amp,recon_amp,multichannel = color)
    return ssim


def generate_calibration_img(slm_res, roi_res, num_circles, circle_radius):
    '''
    Generate calibration image for the SLM
    Args:
        slm_res: tuple, resolution of the SLM
        roi_res: tuple, resolution of the ROI
        num_circles: tuple, number of circles in the horizontal and vertical directions
        circle_radius: int, radius of the circles
    Returns:
        cali_image: np.array, calibration image
    '''
    space_btw_circs = [int(np.ceil(roi / (num_circs - 1))) for roi, num_circs in zip(roi_res, (num_circles[1], num_circles[0]))]
    cali_image = np.zeros(slm_res, dtype=np.float32)
    for i in range(num_circles[0]):
        for j in range(num_circles[1]):
            cv2.circle(cali_image, ((slm_res[1]-roi_res[1])//2+i*space_btw_circs[1], (slm_res[0]-roi_res[0])//2+j*space_btw_circs[0]),circle_radius,(255,255,255),-1)

    return cali_image



def generate_ini_phase(prop_dist, wavelength):
    '''
    This function gis the key function of the Spatial Phase Matching (SPM) method. 
    It generates the initial phase for the input data according to the propagation distance and the wavelength of the light source.
    Args:
        prop_dist: float, propagation distance
        wavelength: float, wavelength of the light source
    Returns:
        ini_phase: float, initial phase
    '''

    ini_phase = -(((10 * cm) % (638 * nm))/(638 * nm) - (prop_dist % wavelength)/wavelength) * 2 * np.pi

    return ini_phase


def generate_ini_phase_test(prop_dist, wavelength):
    ini_phase = (prop_dist - 10*cm) / wavelength * 2 * np.pi

    return ini_phase


def generate_depth_stack(dist_num, depth_map, slm_res, roi_res):
    '''
    Generate depth stack according to the depth map
    Args:
        dist_num: int, number of depth layers
        depth_map: np.array, depth map
        slm_res: tuple, resolution of the SLM
        roi_res: tuple, resolution of the ROI
    Returns:
        mask_3D: list, list of masks for each depth layer
    ''' 

    gray_min = depth_map.min()
    gray_max = depth_map.max()
    gray_delta = (gray_max-gray_min)/dist_num

    mask_3D = []

    padding = ((slm_res[1]-roi_res[1])//2, (slm_res[0]-roi_res[0])//2)

    for k in range(dist_num):
        depth_min = gray_min+k*gray_delta
        depth_max = gray_min+(k+1)*gray_delta
        mask = torch.ones_like(depth_map)
        mask[depth_map<depth_min] = 0.0
        if (k+1)<dist_num:
            mask[depth_map>depth_max] = 0.0

        imaug = Compose([
                        CenterCrop(roi_res),
                        Pad(padding, fill=1, padding_mode='constant'),
                        ])
        
        mask = imaug(mask)
        mask_3D.append(mask)

    return mask_3D


def generate_depth_stack_batch(dist_num, depth_map, slm_res, roi_res, batched_prob_pointer, out_mode='tensor'):
    '''
    This function generates the depth stack for the batched 3D depth layers.
    Args:
        dist_num: int, number of depth layers
        depth_map: torch.tensor, depth map
        slm_res: tuple, resolution of the SLM
        roi_res: tuple, resolution of the ROI
        batched_prob_pointer: torch.tensor, the idxs of the batched data (i.e., [0, 10, 20] for the first, tenth, and twentieth depth layers)
        out_mode: str, output mode, 'list' or 'tensor'
    Returns:
        mask_3D: list or tensor, list of masks for each depth layer
    '''

    gray_min = depth_map.min()
    gray_max = depth_map.max()
    gray_delta = (gray_max-gray_min)/dist_num

    padding = ((slm_res[1]-roi_res[1])//2, (slm_res[0]-roi_res[0])//2)

    imaug = Compose([
                    CenterCrop(roi_res),
                    Pad(padding, fill=0, padding_mode='constant'),
                    ])
    
    imaug_bkg = Compose([
                    CenterCrop(roi_res),
                    Pad(padding, fill=1, padding_mode='constant'),
                    ])

    if out_mode == 'list':
        mask_3D = []

        for k in batched_prob_pointer:
            depth_min = gray_min+k*gray_delta
            depth_max = gray_min+(k+1)*gray_delta
            mask = torch.ones_like(depth_map)
            mask[depth_map<depth_min] = 0.0
            if (k+1)<dist_num:
                mask[depth_map>depth_max] = 0.0

            if k != batched_prob_pointer[-1]:
                mask = imaug(mask)
            else:
                mask = imaug_bkg(mask)
            mask_3D.append(mask)

    elif out_mode == 'tensor':
        depth_min = gray_min+batched_prob_pointer*gray_delta
        depth_max = gray_min+(batched_prob_pointer+1)*gray_delta
        depth_max[-1] = 1000.
        depth_map = depth_map.repeat(1, batched_prob_pointer.size()[0], 1, 1)
        mask = torch.ones_like(depth_map)
        mask[depth_map<depth_min[None, :, None, None]] = 0.0
        mask[depth_map>depth_max[None, :, None, None]] = 0.0
        mask_3D = imaug(mask)
        mask_3D[:, -1, ...] = imaug_bkg(mask[:, -1, ...])
    
    return mask_3D


def prop_dist_cal(z0, dist_num, dz, wavelength):
    ''' calculate propagation distance for each depth layer
    Args:
        z0: float, initial distance of the first layer
        dist_num: int, number of depth layers
        dz: float, distance between layers
        wavelength: float, wavelength of the light source
    Returns:
        prop_dist: np.array, array of propagation distances
    '''

    prop_dist = np.arange(z0, z0+dz*(dist_num-0.5),dz)
    if prop_dist.size != dist_num:
        prop_dist = np.arange(z0, z0+dz*(dist_num),dz)
    prop_dist -= np.mod(prop_dist, wavelength)
    return prop_dist


def my_roll(mat, dim, shifts):
    # assumes 2D array
    n_rows, n_cols, H, W = mat.shape
    
    if dim==0:
        for i in range(n_cols):
            mat[:, i:i+1, ...] = torch.roll(mat[:, i:i+1, ...], shifts[i], dim)

    elif dim==1:
        for i in range(n_rows):
            mat[i:i+1, :, ...] = torch.roll(mat[i:i+1, :, ...], shifts[i], dim)

    return mat