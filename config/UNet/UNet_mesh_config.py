cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

##meta information of the physical settings
meta_info = {
    'Seed' : 20,   # random seed
    'start' : 0,   # the start index of the epoch
    'epoch_number' : 102,   # the total number of the epoch
    'batch_size_train' : 1,   # the batch size during training
    'batch_size_test' : 1,    # the batch size during testing
    'feature_size' : (9.2 * um, 9.2 * um),   # the pixel size of the SLM
    'channel' : 0,   # the channel of the data, 0: Red, 1: Green, 2: Blue
    'wavelength_color' : [638 * nm, 520 * nm, 450 * nm],   # the wavelength of the color channels
    'prop_dist' : 10.5 * cm,   # the distance between the SLM and the object
    'best_loss' : 100.0,     # the initial best loss for checkpoint saving
    'best_psnr' : 0,         # the initial best psnr for checkpoint saving
    'downscale_factor' : 1,  # the downscale factor of the input data, which is used to shuffle the data, 1 means no shuffle
    'epoch_save_fre' : 1,    # the frequency of saving the checkpoint
    'zero_shift' : False,    # whether to zero-shift the zero region if the input data
    'lr_scheduler' : 'cosine',   # the learning rate scheduler
    'calibration' : False,   # whether to generate the calibration hologram
    }


##meta information of the dataset
data = {
        'test_dimension' : '3D',     # the dimension of the data used for evaluation, can be "2D" or "3D"
        
        'data_type' : 'mesh',          # the data format of the data, can be "GS", "mesh", or "LF"
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
        'roi_res' : (1152, 1920),      # the resolution of the ROI
        'data_size' : 20000,          # the size of the training set, whcih is composed of Fourier basis images
        'dist_num': 50,               # the number of the diffraction layers used in layer-based RGBD 3D hologram
        'z0': meta_info['prop_dist'], # the distance between the SLM and the nearest part of the object
        'dz': 0.2*mm,                 # the distance between two adjacent layers
        'feature_size' : meta_info['feature_size'],    # the pixel size of the SLM
        'prop_dist' : meta_info['prop_dist'],    
        'drop_out' : 'around',    # the dropout mode of the input data, can be "around" or "embedded"
        'test_path' : "Test dataset",    # 2D or 3D image dataset used for evaluation
        'filter_x' : 4,    # the filter size of the low-pass filter in the x direction (for Phase Space Tailoring)
        'filter_y' : 4,    # the filter size of the low-pass filter in the y direction (for Phase Space Tailoring)
        'un_zero_blk' : True,    # whether to un-zero the zero region of the input data
        'using_lut' : False,     # whether to use the look-up table to generate the hologram (for point cloud data)
        }


#meta information of the phase-only hologram encoder
model = {
    'name' : 'UNet_hologram',    # the name of the network model (i.e., UNet_hologram, HoloNet)
    'in_channels' : 2*meta_info['downscale_factor']**2,    # the number of the input channels considering the shuffle of the data
    'out_channels' : meta_info['downscale_factor']**2,     # the number of the output channels considering the shuffle of the data
    'depth' : 6,     # the depth of the network
    'initial_feature_num' : 16,     # the initial length of the feature channel
    'norm' : 'batch',               # the normalization method, can be "batch", "weight", and "no" for no normalization
    'up_mode' : 'upconv',           # the up-sampling method, can be "upconv" or "upsample"
    'downscale_factor' : meta_info['downscale_factor'],
    'use_dropout' : False,          # whether to use the dropout layer
    'dropout_pro' : 0.1,            # the probability of the dropout layer
    'forward_mode' : 'PST',         # the forward diffraction method, 
                                    # "ASM": the original angular spectrum method, 
                                    # "PST": the phase space tailoring method,
                                    # "dens": the ASM with modifed (denser) sampling density
    'backward_mode' : 'ASM',        # the backward diffraction method,
    'wei_constrain' : False,        # whether to use the weight constraint during network training
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


# meta information of the calibration hologram
cali = {
        'num_circles': (21, 13),    # the number of circles in the horizontal and vertical directions in the calibration image
        'circle_radius' : 20,       # the radius of the circles in the calibration image
        }


# meta information of the optimizer
optimizer = {
    'name' : 'AdamWOptimizer',     # the name of the optimizer, can be "AdamWOptimizer" or "AdamOptimizer"
    'lr' : 1e-3,                   # the learning rate
    ##meta information of the lr_scheduler
    'lr_scheduler' : {
        'use_lr_scheduler' : False,    # whether to use the learning rate scheduler
        # 'name' : 'CosineWithWarmupScheduler',
        'name' : 'StepLRScheduler',    # the name of the learning rate scheduler, can be "CosineWithWarmupScheduler" or "StepLRScheduler"
        'num_warmup_steps' : 50,       # the number of the warm-up steps
        'num_training_steps' : meta_info['epoch_number'],     
        'num_cycles' : 0.5,            
        'end_lr' : 0.1,
        'step_size' : 25,
        'gamma' : 0.5,
    }
}