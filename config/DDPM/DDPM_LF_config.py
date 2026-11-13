import numpy as np
from torch import nn
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

##meta information of the physical settings
meta_info = {
    'exp_name' : 'DDPM_filter_x4_y4_rectfilter',
    'Seed' : 20,
    'start' : 0,
    'epoch_number' : 102,
    'batch_size_train' : 1,
    'batch_size_test' : 1,
    'feature_size' : (6.4 * um, 6.4 * um),   # the pixel size of the SLM
    'channel' : 0,   # the channel of the data, 0: Red, 1: Green, 2: Blue
    'wavelength_color' : [638 * nm, 520 * nm, 450 * nm],   # the wavelength of the color channels
    'prop_dist' : 9.8 * cm,   # the distance between the SLM and the object
    'best_loss' : 100.0,
    'best_psnr' : 0,
    'downscale_factor' : 1,
    'epoch_save_fre' : 1,
    'no_reload' : False,
    'zero_shift' : False,
    'lr_scheduler' : 'cosine',
    'calibration' : False,
    'test_only' : True,
    }



##meta information of the dataset
data = {
        'test_dimension' : '3D',     # the dimension of the data used for evaluation, can be "2D" or "3D"
        
        'data_type' : 'LF',          # the data format of the data, can be "GS", "mesh", or "LF"
                                     # "GS" means the Guassian splatting based complex hologram,
                                     # "mesh" means the mesh based complex hologram,
                                     # "LF" means the light field based complex hologram
        
        'ini_phase_type' : 'zero',   # set the initial phase, user can choose from (people can also define their own initial phase): 
                                     # pseudo_random, pseudo_random_cons, ramdom, 
                                     # random_cons, zero, FB, FB_cons, FB_constant, 
                                     # pseudo_random_cons_cons, random_cons_constant
        'using_pdf' : False,         # whether to use the probability density function to generate the Fouier basis image
        'conv_h' : 2e4,              # the standard deviation of the probability density function in the height direction
        'conv_w' : 5e4,              # the standard deviation of the probability density function in the width direction
        'frequency_num_range_phase' : (100, 200),    # the range of the number of frequency points used in 2D Fourier basis phase, (min, max)
        'frequency_amp_range_phase' : (0, 1),        # the range of the amplitude of the frequency points used in 2D Fourier basis phase, (min, max)
        'frequency_num_range' : (1, 10),        # the range of the number of frequency points used in 2D Fourier basis amplitude, (min, max)
        'frequency_amp_range' : (0, 1),         # the range of the amplitude of the frequency points used in 2D Fourier basis amplitude, (min, max)
        'slm_res' : (1152, 1920),     # the resolution of the SLM
        'roi_res' : (960, 1620),      # the resolution of the ROI
        'data_size' : 20000,          # the size of the training set, whcih is composed of Fourier basis images
        'dist_num': 50,               # the number of the diffraction layers used in layer-based RGBD 3D hologram
        'z0': meta_info['prop_dist'], # the distance between the SLM and the nearest part of the object
        'dz': 0.1*mm,                 # the distance between two adjacent layers
        'feature_size' : meta_info['feature_size'],    # the pixel size of the SLM
        'prop_dist' : meta_info['prop_dist'],    
        'drop_out' : 'around',    # the dropout mode of the input data, can be "around" or "embedded"
        'test_path' : "Test dataset",    # 2D or 3D image dataset used for evaluation
        'filter_x' : 4,    # the filter size of the low-pass filter in the x direction (for Phase Space Tailoring)
        'filter_y' : 4,    # the filter size of the low-pass filter in the y direction (for Phase Space Tailoring)
        'un_zero_blk' : True,    # whether to un-zero the zero region of the input data
        'using_lut' : True,     # whether to use the look-up table to generate the hologram (for point cloud data)
        }


#meta information of the phase-only hologram encoder
model = {
    'name' : 'DDPM_hologram',
    'kernel_width' : 3,
    'sigma' : 0.,
    'adaptive_phs_shift' : False,
    'axis' : 2,
    'amp_max' : None,
    'phs_max' : np.array([2*np.pi], dtype=np.float32),
    'clamp' : True,
    'normalize' : False,
    "input_dim_ddpm":2,                                        # input dimension of ddpm network
    "output_dim_ddpm":2,                                       # output dimension of ddpm network
    "num_layers_ddpm":8,                                       # number of ddpm network layers
    "num_filters_per_layer_ddpm": 8,                           # number of filters per ddpm network layer
    "filter_width_ddpm":3,                                     # filter width of ddpm network
    "interleave_rate_ddpm":meta_info['downscale_factor'],      # interleaving rate of ddpm network
    "bias_stddev_ddpm": 0.01,                                  # bias standard deviation of ddpm network
    "weight_var_scale_ddpm": 0.25,                             # weight variance of ddpm network   
    "renormalize_input_ddpm": True,                            # normalize input for ddpm network
    "activation_func_ddpm": nn.ReLU(),                         # activation function for intermediate layers of ddpm network
    "output_activation_func_ddpm": nn.Tanh(),                  # activation function for the output layer of ddpm network
    'in_channels' : 2*meta_info['downscale_factor']**2, 
    'out_channels' : meta_info['downscale_factor']**2,
    'downscale_factor' : meta_info['downscale_factor'],
    'forward_mode' : 'PST',
    'backward_mode' : 'ASM',
    'wei_constrain' : False,
    }


## meta information of the model for the neural renderer
model_renderer = {
    'name' : 'UNet_hologram',
    'in_channels' : 2*meta_info['downscale_factor']**2, 
    'out_channels' : 2*meta_info['downscale_factor']**2,
    'depth' : 5,
    'initial_feature_num' : 16,
    'norm' : 'batch',
    'up_mode' : 'upconv',
    'downscale_factor' : meta_info['downscale_factor'],
    'use_dropout' : False,
    'dropout_pro' : 0.1,
    'POH' : False,
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