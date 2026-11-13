import numpy as np
from torch import nn
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

##meta information of the physical settings
meta_info = {
    'exp_name' : 'UNet_DDPM_RI_filter_x4_y4_rectfilter_2w',
    'Seed' : 20,
    'start' : 0,
    'epoch_number' : 102,
    'batch_size_train' : 1,
    'batch_size_test' : 1,
    'feature_size' : (9.2 * um, 9.2 * um),
    'channel' : 0,
    'wavelength_color' : [638 * nm, 520 * nm, 450 * nm],
    'prop_dist' : 10 * cm,
    'best_loss' : 100.0,
    'best_psnr' : 0,
    'downscale_factor' : 1,
    'epoch_save_fre' : 1,
    'no_reload' : False,
    'zero_shift' : False,
    'lr_scheduler' : 'cosine',
    'calibration' : False,
    'test_only' : False,
    }


##meta information of the dataset
data = {
        'test_dimension' : '2D',
        'data_type' : 'layer',
        'using_pdf' : False,
        'conv_h' : 2e4,
        'conv_w' : 5e4,
        'frequency_num_range' : (1, 10),
        'fraquency_amp_range' : (0, 1),
        'slm_res' : (1152, 1920),
        'roi_res' : (768, 1280),
        'data_size' : 20000,
        'dist_num': 50,
        'z0': meta_info['prop_dist'],
        'dz': 0.1*mm,
        'feature_size' : meta_info['feature_size'],
        'wavelength' : meta_info['wavelength_color'][meta_info['channel']],
        'channel' : meta_info['channel'],
        'prop_dist' : meta_info['prop_dist'],
        'drop_out' : 'around',
        'test_path' : "Test DATASET", # 2D or 3D image dataset used for evaluation
        'filter_x' : 4,
        'filter_y' : 4,
        }


#meta information of the phase-only hologram encoder
model = {
    'name' : 'UNet_DPM_hologram',
    'kernel_width' : 3,
    'sigma' : 0.,
    'adaptive_phs_shift' : False,
    'axis' : 2,
    'amp_max' : None,
    'phs_max' : np.array([2*np.pi], dtype=np.float32),
    'clamp' : True,
    'normalize' : False,
    'in_channels' : 2*meta_info['downscale_factor']**2, 
    'out_channels' : 2*meta_info['downscale_factor']**2,
    'depth' : 6,
    'initial_feature_num' : 16,
    'norm' : 'batch',
    'up_mode' : 'upconv',
    'downscale_factor' : meta_info['downscale_factor'],
    'use_dropout' : False,
    'dropout_pro' : 0.1,
    'forward_mode' : 'PST',
    'backward_mode' : 'ASM',
    'wei_constrain' : False,
    }


cali = {
        'num_circles': (21, 13),
        'circle_radius' : 20,
        }


optimizer = {
    'name' : 'AdamWOptimizer',
    'lr' : 1e-3,
    ##meta information of the lr_scheduler
    'lr_scheduler' : {
        'use_lr_scheduler' : False,
        # 'name' : 'CosineWithWarmupScheduler',
        'name' : 'StepLRScheduler',
        'num_warmup_steps' : 50,
        'num_training_steps' : meta_info['epoch_number'],
        'num_cycles' : 0.5,
        'end_lr' : 0.1,
        'step_size' : 25,
        'gamma' : 0.5,
    }
}