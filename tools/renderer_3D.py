import os
import shutil
import torch.nn as nn
from model import propagation_ASM, propagation_ASM_PST, propagation_ASM_dense, propagation_PointSource, propagation_PointSourceRSI, propagation_PointSourceRSII, propagation_RSI, propagation_RSI_encrypted1
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Pad, InterpolationMode
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm, trange
import math
import cv2
import utils as utils
import model.loss as loss
from model.builder import build_net
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


class Renderer():
    '''
    The renderer class for the neural encoder\DPM\AADPM\DDPM
    '''
    def __init__(self, data, model, device, test_only=False, model_complex_generator=None):
        '''Initialize the renderer
        Args:
            data: the meta information of the dataset
            model: the meta information of the neural encoder
            device: the device used for training
            test_only: the flag for testing
        '''
        
        self.test_only = test_only
        self.channel = data['channel']   # define the RGB channel used in the following process, 0: red, 1: green, 2: blue
        self.wei_constrain = model['wei_constrain']   # whether to weighted constrain the reconstructions during loss calculation
        self.feature_size = data['feature_size']  # the pixel size of the SLM, (feature_size, feature_size)
        self.dist_num = data['dist_num']  # the number of diffraction distances
        self.z0 = data['z0']  # the distacne between the SLM and the first diffraction layer
        self.dz = data['dz']  # the distance between two adjacent diffraction layers
        self.wavelength = data['wavelength']  # the wavelength of the light source, Red: 638nm, Green: 520nm, Blue: 450nm
        self.slm_res = data['slm_res']  # the resolution of the SLM
        self.roi_res = data['roi_res']  # the resolution of the ROI
        self.test_dimension = data['test_dimension']  # the dimension of the data used for evaluation, can be "2D" or "3D"
        self.data_type = data['data_type']  # the data format of the data, can be "layer", "point", or "nend-to-end"
        self.filter_x = data['filter_x']  # the filter size of the phase space tailoring in the x direction
        self.filter_y = data['filter_y']  # the filter size of the phase space tailoring in the y direction
        self.using_lut = data['using_lut']  # the flag for using the pre-generated LUT for point source data
        self.padding = ((self.slm_res[1]-self.roi_res[1])//2, (self.slm_res[0]-self.roi_res[0])//2)  # the padding size for the ROI

        self.model_name = model['name']  # the structure of the neural network, 
                                         # can be "UNet_hologram", 
                                         # "UNet_Peng", 
                                         # "UNet_DPM_hologram", 
                                         # "HoloNet",
                                         # more models can be found or defined in the model/net/ folder
        self.downscale_factor = model['downscale_factor']  # the downscale factor for the input shuffling
        self.device = device

        # define the forward and backward propagators
        if model['forward_mode'] == 'ASM':
            self.propagator_forward = propagation_ASM(data)  # the original ASM widely used in the literature
        elif model['forward_mode'] == 'PST':
            self.propagator_forward = propagation_ASM_PST(data)  # the phase space tailoring ASM
        elif model['forward_mode'] == 'dens':
            self.propagator_forward = propagation_ASM_dense(data)  # the ASM with modifed (denser) sampling density
        elif model['forward_mode'] == 'RSI':
            self.propagator_forward = propagation_RSI(data)
        elif model['forward_mode'] == 'RSI_PST':
            self.propagator_forward = propagation_RSI_encrypted1(data)

        if model['backward_mode'] == 'ASM':
            self.propagator_backward = propagation_ASM(data)
        elif model['backward_mode'] == 'PST':
            self.propagator_backward = propagation_ASM_PST(data)
        elif model['backward_mode'] == 'dens':
            self.propagator_backward = propagation_ASM_dense(data)
        elif model['backward_mode'] == 'RSI':
            self.propagator_backward = propagation_RSI(data)
        elif model['backward_mode'] == 'RSI_PST':
            self.propagator_backward = propagation_RSI_encrypted1(data)
        
        # define the point source propagator
        self.propagator_pointsource = propagation_PointSourceRSI(feature_size=self.feature_size,
                                                              channel=self.channel, 
                                                              wavelength=self.wavelength, 
                                                              slm_res=self.slm_res,
                                                              device=self.device)

        self.L1_Loss = nn.L1Loss().to(device)  
        self.L2_Loss = nn.MSELoss().to(device)
        self.SSIM_Loss = loss.SSIM(size_average=False, dim=[-2, -1]).to(device)  # a self-defined SSIM loss function
        self.VGG11_Loss = loss.PerceptualLoss_Vgg11(l1_loss=0.1, lambda_feat=0.025).to(device)  # a perceptual loss function based on resnet

        if (not test_only) or (self.test_dimension == "2D"):
            self.prop_dist = self.z0
            
        else:
            self.prop_dist = utils.prop_dist_cal(self.z0, self.dist_num, self.dz, self.wavelength)  # if the data is 3D, calculate the propagation distances
            self.prop_dist = torch.from_numpy(self.prop_dist).unsqueeze(0).type(torch.float32)

        self.init_propagator(self.prop_dist)  # initialize the propagators

        if self.data_type == "end-to-end":
            self.init_end_to_end(model_complex_generator)


    def init_end_to_end(self, model_complex_generator):
        '''Initialize the end-to-end complex hologram generator
        '''
        self.neural_complex_generator = build_net(model_complex_generator)
        self.neural_complex_generator.to(self.device)
        # Load checkpoints
        check_point_dir = os.path.join(os.getcwd(), "WH", "complex_generator_{}nm".format(int(self.wavelength/nm)), "checkpoints", "Best_Loss.pth")
        
        print('Reloading the neural complex generator from', check_point_dir)
        ckpt = torch.load(check_point_dir)

        self.neural_complex_generator.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['network_fn_state_dict'].items()})

        self.neural_complex_generator.eval()

        print('Reloading has been Done!')

        self.precomputed_H_z0 = self.propagator_backward(u_in=torch.empty((1, 1, *self.slm_res), dtype=torch.complex64), 
                                                z=-self.prop_dist[0, 0].item(), return_H=True, 
                                                filter_x=self.filter_x, filter_y=self.filter_y)
        self.precomputed_H_z0 = self.precomputed_H_z0.to(self.device)


    def init_propagator(self, prop_dist):
        '''Initialize the propagators
        including the forward and backward propagators and the layer-wise propagators,
        the Transfer Functions H and H_inverse are precomputed and matained in the memory
        Args:
            prop_dist: the propagation distances for each depth layer
        '''

        # precompute the Transfer Functions H and H_inverse for the forward and backward propagators
        self.precomputed_H = self.propagator_forward(u_in=torch.empty((1, 1, *self.slm_res), dtype=torch.complex64), 
                                                    z=prop_dist, return_H=True, 
                                                    filter_x=self.filter_x, filter_y=self.filter_y) # [1, depth_num, slm_height, slm_width]
        
        self.precomputed_H_inverse = self.propagator_backward(u_in=torch.empty((1, 1, *self.slm_res), dtype=torch.complex64), 
                                                            z=-prop_dist, return_H=True,
                                                            filter_x=self.filter_x, filter_y=self.filter_y) # [1, depth_num, slm_height, slm_width]
        
        if self.test_dimension == '3D':
            # precompute the Transfer Functions H between adjacent depth layers for the silhouette layer propagation
            self.H_layer_inverse = self.propagator_backward(u_in=torch.empty((1, 1, *self.slm_res), dtype=torch.complex64), 
                                                            z=-(prop_dist[:, 1:]-prop_dist[:, 0:-1]), return_H=True,
                                                            filter_x=self.filter_x, filter_y=self.filter_y) # [1, 1, slm_height, slm_width]
            if self.H_layer_inverse.size()[1] == 1:
                self.H_layer_inverse = self.H_layer_inverse[:, 0:1, ...].to(self.device)
        
        # during training and 2D evaluation, the precomputed Transfer Functions are stored in the GPU memory,
        # otherwise, they are stored in the CPU memory, and moved to the GPU memory when needed
        if self.precomputed_H.size()[1] == 1:
            self.precomputed_H = self.precomputed_H.to(self.device)
            self.precomputed_H_inverse = self.precomputed_H_inverse.to(self.device)


    def layer_prop(self, net, image, ini_phase, cali=False):
        '''Layer-wise propagation for 2D image data
        Args:
            net: the neural encoder
            image: the input image, shape: (batch_size, dist_num, slm_height, slm_width)
            ini_phase: the initial phase, shape: (batch_size, 1, slm_height, slm_width)
            cali: the flag for calibration
        Returns:
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        '''

        if not net.training:    # during testing and evaluation, Spatial Phase Matching is adopted to modified the initial phase of each layer element
            phase_shift = utils.generate_ini_phase(self.prop_dist, self.wavelength)  # Spatial Phase Matching according to the propagation distances and wavelength
            if torch.is_tensor(phase_shift):
                phase_shift = phase_shift.type(torch.float32).to(image.device)
                obj_real, obj_imag = utils.polar_to_rect(image, phase_shift[:, :, None, None] * torch.ones_like(image) + ini_phase)
            else:
                obj_real, obj_imag = utils.polar_to_rect(image, phase_shift * torch.ones_like(image) + ini_phase)
        else:
            obj_real, obj_imag = utils.polar_to_rect(image, ini_phase)  # during training, the initial phase is fixed

        object_field = torch.complex(obj_real, obj_imag)  # [batch_size, dist_num, slm_height, slm_width]
        
        if self.precomputed_H_inverse.size()[1] > 1:  # for 3D data, the layer-wise propagation is adopted
            interval = 2  # calculate the propagation of two adjacent depth layers at a time
            input_field = torch.zeros_like(object_field) # [batch_size, dist_num, slm_height, slm_width]
            for i in range(self.precomputed_H_inverse.size()[1] // interval):

                precomputed_H_inverse = self.precomputed_H_inverse[:, i*interval:(i+1)*interval,...].to(self.device)
                input_field[:, i*interval:(i+1)*interval,...] = self.propagator_backward(u_in=object_field[:, i*interval:(i+1)*interval,...], 
                                                                                         z=-self.prop_dist[:, i*interval:(i+1)*interval],
                                                                                         precomped_H=precomputed_H_inverse)
        else:  # for 2D data, only one propagation needed
            input_field = self.propagator_backward(u_in=object_field, z=-self.prop_dist, 
                                                    dtype=torch.float32, precomped_H=self.precomputed_H_inverse)

        if not cali:
            input_field = torch.sum(input_field, dim=1, keepdim=True)  # for layer-based hologram generation, the complex hologram is the sum of all the layer elements
        else:
            input_field = input_field.permute(1, 0, 2, 3)  # for calibration, the complex hologram is permuted for the calibration process

        return input_field
    

    def silhouette_layer_prop(self, net, image, depth_map, ini_phase, cali):
        '''Silhouette layer-wise propagation for 3D RGBD data
           Yan et al., Generating Multi‐Depth 3D Holograms Using a Fully Convolutional Neural Network, 2024.
           Zhang et al., Computer-generated hologram with occlusion effect using layer-based processing, 2017.
        Args:
            net: the neural encoder
            image: the input image, shape: (batch_size, dist_num, slm_height, slm_width)
            depth_map: the depth map, shape: (batch_size, 1, slm_height, slm_width)
            ini_phase: the initial phase, shape: (batch_size, 1, slm_height, slm_width)
            cali: the flag for calibration
        Returns:
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        '''

        if not net.training:
            phase_shift = utils.generate_ini_phase(self.prop_dist, self.wavelength)  # Spatial Phase Matching according to the propagation distances and wavelength
            if torch.is_tensor(ini_phase):
                phase_shift = phase_shift.type(torch.float32).to(image.device)
                obj_real, obj_imag = utils.polar_to_rect(image, phase_shift[:, :, None, None] * torch.ones_like(image) + ini_phase)
            else:
                obj_real, obj_imag = utils.polar_to_rect(image, phase_shift * torch.ones_like(image) + ini_phase)
        else:
            obj_real, obj_imag = utils.polar_to_rect(image, ini_phase)

        object_field = torch.complex(obj_real, obj_imag)  # [batch_size, dist_num, slm_height, slm_width]

        depth_mask = utils.generate_depth_stack_batch(self.dist_num, depth_map, 
                                                      self.slm_res, self.roi_res, 
                                                      torch.range(0, self.dist_num-1).to(self.device), 'tensor')  # generate the depth mask stack,
                                                                                                                  # which is needed for the silhouette layer-wise propagation

        # input_field = self.propagator_backward(u_in=object_field, z=-self.prop_dist, feature_size=self.feature_size, wavelength=self.wavelength,
        #                             dtype=torch.float32, precomped_H=self.precomputed_H_inverse)
        if self.precomputed_H_inverse.size()[1] > 1:
            # input_field = object_field[:, -1:, ...]
            input_field = torch.sum(object_field, dim=1, keepdim=True)  # the first layer element is the sum of all the layer elements
            imaug = Compose([
                        CenterCrop(self.roi_res),
                        Pad(self.padding, fill=1, padding_mode='constant'),
                        ])
            M_n = 1 - depth_mask
            for i in reversed(range(self.precomputed_H_inverse.size()[1])):  # the silhouette layer-wise propagation is adopted for each layer
                if i != 0:
                    H_layer_inverse = self.H_layer_inverse[:, (i-1):i, ...].to(self.device)
                    C_n = self.propagator_backward(u_in=input_field,
                                                z=-(self.prop_dist[:, i:(i+1)]-self.prop_dist[:, (i-1):i]),
                                                precomped_H=H_layer_inverse)
                
                    L_n = C_n * M_n[:, i-1:i,...]
                    input_field = L_n + object_field[:, i-1:i, ...]
                else:
                    input_field = self.propagator_backward(u_in=input_field,
                                                            z=-self.prop_dist[:, i:(i+1)],
                                                            precomped_H=self.precomputed_H_inverse[:, i:(i+1), ...].to(self.device))

        else:
            input_field = self.propagator_backward(u_in=object_field, z=-self.prop_dist, 
                                                    dtype=torch.float32, precomped_H=self.precomputed_H_inverse)

        if cali:
            input_field = input_field.permute(1, 0, 2, 3)
        

        return input_field
    

    def point_prop(self, image, depth_map):
        '''Point source propagation for 3D point cloud data
        Args:
            image: the input image, shape: (batch_size, 1, slm_height, slm_width)
            depth_map: the depth map, shape: (batch_size, 1, slm_height, slm_width)
        Returns:
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        '''

        W = self.propagator_pointsource.W  # the x coordinates of the point sources, [1, slm_height, slm_width]
        H = self.propagator_pointsource.H  # the y coordinates of the point sources, [1, slm_height, slm_width]

        z_span = self.dz * self.dist_num  # the total propagation range of the point sources

        norm_depth = (depth_map - torch.min(depth_map)) / (torch.max(depth_map) - torch.min(depth_map))  # normalize the depth map to [0, 1], [batch_size, 1, slm_height, slm_width]
        Z = norm_depth * z_span + self.z0  # the z coordinates of the point sources, [batch_size, 1, slm_height, slm_width]

        coor_matrix = torch.concat([W.unsqueeze(1), H.unsqueeze(1), Z], dim = 1)  # the coordinates of the point sources, [batch_size, 3, slm_height, slm_width]
        # coor_matrix = utils.crop_image(coor_matrix, target_shape=self.roi_res, stacked_complex=False)

        coor_matrix = coor_matrix.permute(0, 2, 3, 1)  # [batch_size, slm_height, slm_width, 3]
        coor_vector = torch.reshape(coor_matrix, (coor_matrix.size()[0], -1, coor_matrix.size()[-1]))  # [batch_size, slm_height*slm_width, 3]

        # image = utils.crop_image(image, target_shape=self.roi_res, stacked_complex=False)
        amp_vector = torch.reshape(image, (image.size()[0], -1))  # [batch_size, slm_height*slm_width]

        input_field = torch.zeros_like(depth_map, dtype=torch.complex64)  # [batch_size, 1, slm_height, slm_width]

        batch_size = 9  # the batch size for the point source propagation, can be adjusted according to the GPU memory

        for i in trange(coor_vector.size()[1] // batch_size):
            
            # the point source propagation is divided into several batches to accelerate the computation,
            # the Spatial Phase Matching is realized in the Class: propagation_PointSourceRSI
            batch_field, _ = self.propagator_pointsource(u_in=amp_vector[:, i*batch_size:(i+1)*batch_size], coor=coor_vector[:, i*batch_size:(i+1)*batch_size, :])
            batch_field = torch.sum(batch_field, dim=1, keepdim=True)
            input_field = input_field + batch_field


        return input_field
    


    def point_prop_lut(self, lut_path):
        '''Point source propagation for 3D point cloud data using pre-generated LUT
           as the point source propagation is time-consuming, the pre-generated LUT can be used for acceleration
        Args:
            lut_path: the path of the pre-generated LUT
        Returns:
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        '''

        input_field = np.load(lut_path)
        input_field = torch.from_numpy(input_field).to(self.device)

        return input_field
    

    def end_to_end_generator(self, image, depth_map):
        '''Generate a complex hologram for the input 3D RGBD data using the neural complex generator
        Args:
            image: the input image for a single RGB channel, shape: (batch_size, 1, H, W)
            depth_map: the depth map, shape: (batch_size, 1, H, W)
        Returns:
            spectrum_field: the complex hologram, shape: (batch_size, 1, H, W)
        '''

        input_field = torch.cat([image, depth_map], dim=1)
        input_unshuffle = F.pixel_unshuffle(input_field, self.downscale_factor)     #(B, C*r**2, H//r, W//r)

        print("Generating the hologram using the neural complex generator...")
        spectrum_shuffle = self.neural_complex_generator(input_unshuffle)          #(B, 2*r**2, H//r, W//r), the two channels are real and imaginary parts respectively
        print("Generating has been Done!")
        spectrum = F.pixel_shuffle(spectrum_shuffle, self.downscale_factor)          #(B, 2, H, W)
        
        spectrum_field = torch.complex(spectrum[:, 0:1, ...], spectrum[:, 1:2, ...])

        spectrum_field_preprop = self.propagator_backward(u_in=spectrum_field, z=-self.z0, 
                                                    dtype=torch.float32, precomped_H=self.precomputed_H_z0)

        return spectrum_field_preprop


    def field_reconstructor(self, slm_field):
        '''Reconstruct the wavefield from the hologram
        Args:
            slm_field: the complex amplitude on the SLM, shape: (batch_size, 1, slm_height, slm_width)
        Returns:
            recon_field: the reconstruction from the hologram, shape: (batch_size, depth_num, slm_height, slm_width)
        '''

        if self.precomputed_H.size()[1] > 1:
            interval = 2  # calculate the propagation of two adjacent depth layers at a time, can be adjusted according to the GPU memory
            recon_field = torch.zeros_like(slm_field).repeat(1, self.dist_num, 1, 1)  # [batch_size, depth_num, slm_height, slm_width]
            for i in range(self.precomputed_H.size()[1] // interval):

                precomputed_H = self.precomputed_H[:, i*interval:(i+1)*interval,...].to(self.device)
                recon_field[:, i*interval:(i+1)*interval,...] = self.propagator_forward(u_in=slm_field, 
                                                                                         z=self.prop_dist[:, i*interval:(i+1)*interval],
                                                                                         precomped_H=precomputed_H)
        else:
            recon_field = self.propagator_forward(u_in=slm_field, z=self.prop_dist, 
                                            dtype=torch.float32, precomped_H=self.precomputed_H)
            
        return recon_field


    def predict(self, net, image, ini_phase, depth_map=None, cali=False):
        '''Predict the phase-only hologram from the input data (2D image, RGBD 3D image or 3D point cloud)
        Args:
            net: the neural encoder
            image: the input image, shape: (batch_size, dist_num, slm_height, slm_width)
            ini_phase: the initial phase, shape: (batch_size, 1, slm_height, slm_width)
            depth_map: the depth map, shape: (batch_size, 1, slm_height, slm_width)
            cali: the flag for calibration
        Returns:
            recon_field: the reconstruction from phase-only hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            target_recon_field: the target reconstructed from complex hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            slm_phase: the phase-only hologram, shape: (batch_size, 1, slm_height, slm_width)
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        ''' 

        # generate the complex hologram according to the data type
        if self.data_type == 'layer':
            if self.test_dimension == '3D':
                # silhouette layer propagation for 3D RGBD data
                # Yan et al., Generating Multi‐Depth 3D Holograms Using a Fully Convolutional Neural Network, 2024.
                # Zhang et al., Computer-generated hologram with occlusion effect using layer-based processing, 2017.
                input_field = self.silhouette_layer_prop(net=net, image=image, depth_map=depth_map, ini_phase=ini_phase, cali=cali)
            else:
                # layer propagation for 2D image
                input_field = self.layer_prop(net=net, image=image, ini_phase=ini_phase, cali=cali)
        elif self.data_type == 'point':
            if not self.using_lut:
                # point source propagation for 3D point cloud
                input_field = self.point_prop(image=image, depth_map=depth_map)
            else:
                # point source propagation for 3D point cloud using pre-generated LUT
                input_field = self.point_prop_lut(lut_path=os.path.join(os.getcwd(), 
                                                                        "WH", 
                                                                        "UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w", 
                                                                        "results",
                                                                        "point_10cm_638nm_P44.70_S0.99",
                                                                        "0",
                                                                        "complex hologram",
                                                                        "6_complex_hologram_epoch_081_No005.npy"))
        elif self.data_type == 'end-to-end':
            input_field = self.end_to_end_generator(image=image, depth_map=depth_map)

        # normalize the complex hologram according to its amplitude
        if not self.using_lut:
            input_amp = input_field.abs()
            input_field = (input_field - torch.min(input_amp))/(torch.max(input_amp) - torch.min(input_amp))

        # concatenate the real and imaginary parts of the complex hologram before inputting into the neural encoder
        input_real = input_field.real
        input_img = input_field.imag
        input_cat = torch.cat([input_real, input_img], 1) # [batch_size, 2, slm_height, slm_width]

        # (optional) shuffle the input data to alleviate the memory consumption
        # by default, the input data is unshuffled [with the downscale factor of 1]
        input_unshuffle = F.pixel_unshuffle(input_cat, self.downscale_factor) # [batch_size, 2*downscale_factor^2, slm_height//downscale_factor, slm_width//downscale_factor]

        # predict the phase-only hologram
        slm_phase_shuffle = net(input_unshuffle) # [batch_size, downscale_factor^2, slm_height//downscale_factor, slm_width//downscale_factor]
        # shuffle the predicted phase-only hologram back to the original resolution
        slm_phase = F.pixel_shuffle(slm_phase_shuffle, self.downscale_factor) # [batch_size, 1, slm_height, slm_width]
        
        # convert the phase-only hologram to the complex hologram with the constant amplitude
        # this step is necessary for the quantitative evaluation
        slm_real, slm_imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(slm_real, slm_imag)

        # reconstruct the wavefield from phase-only hologram
        recon_field = self.field_reconstructor(slm_field) # [batch_size, depth_num, slm_height, slm_width]
        # reconstruct the wavefield from complex hologram
        target_recon_field = self.field_reconstructor(input_field) # [batch_size, depth_num, slm_height, slm_width]


        return recon_field, target_recon_field, slm_phase, input_field

    def predict_other3D(self, net, data, ini_phase, cali=False):
        '''Predict the phase-only hologram from the input data (GS, mesh, or LF)
        Args:
            net: the neural encoder
            data: the input data, shape: (batch_size, 1, slm_height, slm_width), torch.complex64
            ini_phase: the initial phase, shape: (batch_size, 1, slm_height, slm_width)
            cali: the flag for calibration
        Returns:
            recon_field: the reconstruction from phase-only hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            target_recon_field: the target reconstructed from complex hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            slm_phase: the phase-only hologram, shape: (batch_size, 1, slm_height, slm_width)
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        ''' 

        # generate the complex hologram according to the data type
        input_field = data

        # normalize the complex hologram according to its amplitude
        if not self.using_lut:
            input_amp = input_field.abs()
            input_field = (input_field - torch.min(input_amp))/(torch.max(input_amp) - torch.min(input_amp))

        # concatenate the real and imaginary parts of the complex hologram before inputting into the neural encoder
        input_real = input_field.real
        input_img = input_field.imag
        input_cat = torch.cat([input_real, input_img], 1) # [batch_size, 2, slm_height, slm_width]

        # (optional) shuffle the input data to alleviate the memory consumption
        # by default, the input data is unshuffled [with the downscale factor of 1]
        input_unshuffle = F.pixel_unshuffle(input_cat, self.downscale_factor) # [batch_size, 2*downscale_factor^2, slm_height//downscale_factor, slm_width//downscale_factor]

        # predict the phase-only hologram
        slm_phase_shuffle = net(input_unshuffle) # [batch_size, downscale_factor^2, slm_height//downscale_factor, slm_width//downscale_factor]
        # shuffle the predicted phase-only hologram back to the original resolution
        slm_phase = F.pixel_shuffle(slm_phase_shuffle, self.downscale_factor) # [batch_size, 1, slm_height, slm_width]
        
        # convert the phase-only hologram to the complex hologram with the constant amplitude
        # this step is necessary for the quantitative evaluation
        slm_real, slm_imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(slm_real, slm_imag)

        # reconstruct the wavefield from phase-only hologram
        recon_field = self.field_reconstructor(slm_field) # [batch_size, depth_num, slm_height, slm_width]
        # reconstruct the wavefield from complex hologram
        target_recon_field = self.field_reconstructor(input_field) # [batch_size, depth_num, slm_height, slm_width]

        return recon_field, target_recon_field, slm_phase, input_field


    def predict_parallax(self, data, slm_phase, input_field):
        '''Predict the motion parallax effect from the phase-only hologram and the complex hologram'''

        slm_real, slm_imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(slm_real, slm_imag)

        recon_parallax_field = self.generate_stabilized_parallax_final(hologram=slm_field.squeeze(), device=slm_field.device)

        target_parallax_field = self.generate_stabilized_parallax_final(hologram=input_field.squeeze(), device=input_field.device)

        return recon_parallax_field, target_parallax_field

    def generate_stabilized_parallax_final(
            self,
            hologram, 
            recon_distance=0.095,     # Zero parallax plane distance. Objects at this distance will be completely stationary.
            aperture_radius_ratio=0.9, # Increase this value to reduce depth of field and make parallax more obvious. Recommended range 0.5 - 0.9
            view_range_ratio=0.5,   # Scanning range
            num_views=3,           # Number of views (odd number to include the center view)
            pad_factor=1.0,         # Padding factor, 1.0 means padding to 2 times the original size to prevent wrapping
            device='cuda',
        ):
        
        # 1. Load and significantly pad the hologram
        H_orig, W_orig = hologram.shape

        pad_h = int(H_orig * pad_factor)
        pad_w = int(W_orig * pad_factor)
        
        # Pad with zeros (physically equivalent to blocking the surroundings with opaque material)
        h_padded = F.pad(hologram.unsqueeze(0).unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0).squeeze()
        H_pad, W_pad = h_padded.shape
        
        # 2. Precompute spectrum and coordinate grid
        spectrum = torch.fft.fftshift(torch.fft.fft2(h_padded))
        Y, X = torch.meshgrid(torch.arange(H_pad, device=device), torch.arange(W_pad, device=device), indexing='ij')
        cy, cx = H_pad // 2, W_pad // 2
        
        # 3. Calculate frequency and angle parameters
        df_y = 1.0 / (H_pad * self.feature_size[0])
        df_x = 1.0 / (W_pad * self.feature_size[1])
        
        # Calculate scanning range based on original size
        max_shift_y = int((H_orig // 2) * view_range_ratio)
        max_shift_x = int((W_orig // 2) * view_range_ratio)
        
        shift_y_list = np.linspace(-max_shift_y, max_shift_y, num_views).astype(int)
        shift_x_list = np.linspace(-max_shift_x, max_shift_x, num_views).astype(int)
        
        # Aperture size (determined based on original size to keep physical meaning consistent)
        aperture_r_px = int(min(H_orig, W_orig) * aperture_radius_ratio)

        LF_recon = torch.zeros((num_views, num_views, self.roi_res[0], self.roi_res[1]), device=device)

        for i, dy in enumerate(shift_y_list):
            for j, dx in enumerate(shift_x_list):
                
                # --- A. Frequency domain selection (simulating eye looking at a specific angle) ---
                center_y = cy + dy
                center_x = cx + dx
                
                # Use Gaussian Soft Aperture to avoid ringing artifacts
                dist_sq = (Y - center_y)**2 + (X - center_x)**2
                sigma = aperture_r_px / 2.0 # Control Gaussian width
                mask = torch.exp(-dist_sq / (2 * sigma**2))
                
                filtered_spectrum = spectrum * mask
                
                # --- B. Carrier Demodulation ---
                # Move the selected spectrum region back to the center. This is equivalent to "straightening" the tilted view.
                filtered_spectrum = torch.roll(filtered_spectrum, shifts=(-dy, -dx), dims=(0, 1))
                
                # --- C. Return to spatial domain and propagate ---
                # IFFT to get complex amplitude on the hologram plane
                view_hologram_at_0 = torch.fft.ifft2(torch.fft.ifftshift(filtered_spectrum))
                
                # ASM propagation to the specified reconstruction distance
                # This step produces perspective effect: objects at distance z will shift by z * tan(theta)
                recon_at_z = self.propagator_forward(u_in=view_hologram_at_0.unsqueeze(0).unsqueeze(0), z=recon_distance).squeeze()
                intensity = torch.abs(recon_at_z)
                
                # --- D. [Core] Precise Geometric Stabilization Compensation ---
                # 1. Calculate the true physical angle sin(theta) corresponding to the current frequency
                fy_phys = dy * df_y
                fx_phys = dx * df_x
                sin_theta_y = np.clip(self.wavelength * fy_phys, -1.0, 1.0) # Prevent numerical errors causing out of bounds
                sin_theta_x = np.clip(self.wavelength * fx_phys, -1.0, 1.0)
                
                # 2. Calculate accurate tan(theta) = sin / sqrt(1-sin^2)
                # Avoid division by zero error (although tan is 0 when sin=0 at center view, just for safety)
                tan_theta_y = sin_theta_y / np.sqrt(1 - sin_theta_y**2 + 1e-9)
                tan_theta_x = sin_theta_x / np.sqrt(1 - sin_theta_x**2 + 1e-9)
                
                # 3. Calculate physical geometric shift
                # When the view tilts positively (right/down), the object moves negatively (left/up) on the image plane
                # Shift_img = - z * tan(theta)
                geom_shift_y_m = - recon_distance * tan_theta_y
                geom_shift_x_m = - recon_distance * tan_theta_x
                
                # 4. Convert to pixel shift
                shift_y_px = int(round(geom_shift_y_m / self.feature_size[0]))
                shift_x_px = int(round(geom_shift_x_m / self.feature_size[1]))

                # 5. Apply compensation (reverse shift image)
                # We need to "pull back" the shifted image, so the roll direction is opposite to the object movement direction
                # Roll shift = - (Shift_img) = + z * tan(theta)
                # Here shift_x_px is already negative, so we should roll (-shift_x_px) which is positive
                intensity_stabilized = torch.roll(intensity, shifts=(-shift_y_px, -shift_x_px), dims=(0, 1))
                
                # --- E. Center crop (remove padding area) ---
                # After rolling, the target object returns to the center of the padded image
                # Directly crop the center region of the original size
                y_start = pad_h
                x_start = pad_w
                img_cropped = intensity_stabilized[y_start:y_start+H_orig, x_start:x_start+W_orig]

                # Define the region of interest (ROI) for the light field
                roi_start_h = (self.slm_res[0] - self.roi_res[0]) // 2
                roi_start_w = (self.slm_res[1] - self.roi_res[1]) // 2 + 90
                roi_height = self.roi_res[0]
                roi_width = self.roi_res[1]
                img_roi = img_cropped[roi_start_h:roi_start_h+roi_height, roi_start_w:roi_start_w+roi_width]

                LF_recon[i, j] = img_roi

        return LF_recon


    def image_preprocessor(self, image):
        '''Preprocess the input image for the loss calculation
           if the weighted constrain is adopted, the input image is cropped to the ROI region
        Args:
            image: the input image, shape: (batch_size, depth_num, slm_height, slm_width)
        Returns:
            image_loss: the preprocessed image for the loss calculation, shape: (batch_size, 1, roi_height, roi_width)
        '''

        image_wei = utils.crop_image(image, target_shape=self.roi_res, stacked_complex=False)  # image cropper

        if self.wei_constrain:
            image_loss = image_wei

        else:
            image_loss = image

        return image_loss
    

    def image_normalizer(self, image, target):
        '''Normalize the input image according to the target image, 
           this step is necessary for the quantitative evaluation, 
           the normalization is adopted for each channel respectively
        Args:
            image: the input image, shape: (batch_size, dist_num, slm_height, slm_width)
            target: the target image, shape: (batch_size, dist_num, slm_height, slm_width)
        Returns:
            image: the normalized image, shape: (batch_size, dist_num, slm_height, slm_width)
        '''

        image_mean = torch.mean(image, dim=[-1,-2], keepdim=True)  # [batch_size, dist_num, 1, 1]
        target_mean = torch.mean(target, dim=[-1,-2], keepdim=True)  # [batch_size, dist_num, 1, 1]
        if all(image_mean[0, :, 0, 0] != 0) and all(target_mean[0, :, 0, 0] != 0):
            image = image*(target_mean/image_mean) 

        return image


    def evaluate(self, 
                 target, 
                 recon_field, 
                 target_recon_field,
                 depth_mask=None,
                 train=True,
                 cali=False):
        '''Evaluate the performance of the neural encoder
        Args:
            target: the target image, shape: (batch_size, 1, slm_height, slm_width)
            recon_field: the reconstruction from phase-only hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            target_recon_field: the target reconstructed from complex hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            depth_mask: the depth mask stack for 3D image, shape: (batch_size, depth_num, slm_height, slm_width)
            train: the flag for training
            cali: the flag for calibration
        Returns:
            lossValue: the loss value
            PSNR: the peak signal-to-noise ratio of the reconstructions
            SSIM: the structural similarity index of the reconstructions
        '''

        lossValue = 0

        recon_amp = recon_field.abs() # [batch_size, depth_num, slm_height, slm_width]
        target_recon_amp = target_recon_field.abs()
        
        # joint the focused parts of the reconstructions at each diffraction distance for 3D image
        if self.test_dimension == "3D" and (not cali):
            joint_recon_amp = torch.sum(recon_amp * depth_mask, dim=1, keepdim=True) # [batch_size, 1, slm_height, slm_width]
            joint_target_recon_amp = torch.sum(target_recon_amp * depth_mask, dim=1, keepdim=True)

        if cali:
            target = target.repeat(1, recon_amp.size()[1], 1, 1)

        # preprocess the reconstructions and the target image (crop and resize)
        recon_amp_loss = self.image_preprocessor(recon_amp)
        target_loss = self.image_preprocessor(target)

        # normalize the reconstructions according to the target image for evaluation
        if not train:
            recon_amp_loss = self.image_normalizer(image=recon_amp_loss, target=target_loss)

        # calculate the loss value with the L1 and L2 loss functions
        lossValue = self.L1_Loss(recon_amp_loss, target_loss) + \
                        10 * self.L2_Loss(recon_amp_loss, target_loss)
        
        if torch.isnan(lossValue):
            raise ValueError("Zero division error!")
        
        # evaluate the reconstructions with the PSNR and SSIM metrics
        with torch.no_grad():
            
            # crop the ROI region for evaluation
            recon_amp_eval = utils.crop_image(recon_amp, target_shape=self.roi_res, stacked_complex=False) # [batch_size, depth_num, roi_height, roi_width]
            target_recon_amp_eval =  utils.crop_image(target_recon_amp, target_shape=self.roi_res, stacked_complex=False) # [batch_size, depth_num, roi_height, roi_width]

            # crop the joint image for 3D recontructions
            if self.test_dimension == "3D" and (not cali):
                joint_recon_amp_eval =  utils.crop_image(joint_recon_amp, target_shape=self.roi_res, stacked_complex=False) # [batch_size, 1, roi_height, roi_width]
                joint_target_recon_amp_eval =  utils.crop_image(joint_target_recon_amp, target_shape=self.roi_res, stacked_complex=False) # [batch_size, 1, roi_height, roi_width]
            target_eval = utils.crop_image(target, target_shape=self.roi_res, stacked_complex=False)

            # normalize the reconstructions
            if self.test_dimension == '2D':
                recon_amp_eval = self.image_normalizer(image=recon_amp_eval, target=target_eval)
            elif self.test_dimension == '3D':
                recon_amp_eval = self.image_normalizer(image=recon_amp_eval, target=target_recon_amp_eval)

            if self.test_dimension == "3D" and (not cali):
                joint_recon_amp_eval_joint = self.image_normalizer(image=joint_recon_amp_eval, target=joint_target_recon_amp_eval)
                joint_recon_amp_eval_target = self.image_normalizer(image=joint_recon_amp_eval, target=target_eval)

            # calculate the PSNR and SSIM metrics
            if self.test_dimension == '2D':
                PSNR_layer = utils.psnr2(recon_amp_eval, target_eval).squeeze().tolist()
                SSIM_layer = utils.ssim2(recon_amp_eval, target_eval)
            elif self.test_dimension == '3D':
                PSNR_layer = utils.psnr2(recon_amp_eval, target_recon_amp_eval).squeeze().tolist()
                if target_recon_amp_eval.size()[1] > 1:  
                    # if the tensor has multiple channels, calculate the SSIM for each channel using a self-defined function
                    SSIM_layer = self.SSIM_Loss(recon_amp_eval.detach().cpu(), target_recon_amp_eval.detach().cpu()).squeeze().tolist()
                else:
                    SSIM_layer = self.SSIM_Loss(recon_amp_eval, target_recon_amp_eval).squeeze().tolist()

            if self.test_dimension == "3D" and (not cali):
                PSNR_joint = utils.psnr2(joint_recon_amp_eval_joint, joint_target_recon_amp_eval).squeeze().tolist()
                SSIM_joint = self.SSIM_Loss(joint_recon_amp_eval_joint, joint_target_recon_amp_eval).squeeze().tolist()

                PSNR_target = utils.psnr2(joint_recon_amp_eval_target, target_eval).squeeze().tolist()
                SSIM_target = self.SSIM_Loss(joint_recon_amp_eval_target, target_eval).squeeze().tolist()

            else:
                PSNR_joint = PSNR_layer
                SSIM_joint = SSIM_layer
                PSNR_target = PSNR_layer
                SSIM_target = SSIM_layer

            # convert the PSNR and SSIM metrics to the dictionary format
            PSNR_layer = PSNR_layer if isinstance(PSNR_layer, list) else [PSNR_layer]
            SSIM_layer = SSIM_layer if isinstance(SSIM_layer, list) else [SSIM_layer]

            PSNR = {
                'PSNR_layer' : PSNR_layer,
                'PSNR_joint' : PSNR_joint,
                'PSNR_target' : PSNR_target,
            }

            SSIM = {
                'SSIM_layer' : SSIM_layer,
                'SSIM_joint' : SSIM_joint,
                'SSIM_target' : SSIM_target,
            }

        return lossValue, PSNR, SSIM


    def evaluate_other3D(self,
                        recon_field,
                        target_recon_field,
                        train=True,
                        cali=False):
        '''Evaluate the performance of the neural encoder for other 3D data (GS, mesh, or LF)
        Args:
            recon_field: the reconstruction from phase-only hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            target_recon_field: the target reconstructed from complex hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            train: the flag for training
            cali: the flag for calibration
        Returns:
            PSNR: the peak signal-to-noise ratio of the reconstructions
            SSIM: the structural similarity index of the reconstructions
        '''
        
        # evaluate the reconstructions with the PSNR and SSIM metrics
        with torch.no_grad():

            recon_amp = recon_field.abs() # [batch_size, depth_num, slm_height, slm_width]
            target_recon_amp = target_recon_field.abs()
            
            # crop the ROI region for evaluation
            recon_amp_eval = utils.crop_image(recon_amp, target_shape=self.roi_res, stacked_complex=False) # [batch_size, depth_num, roi_height, roi_width]
            target_recon_amp_eval =  utils.crop_image(target_recon_amp, target_shape=self.roi_res, stacked_complex=False) # [batch_size, depth_num, roi_height, roi_width]

            recon_amp_eval = self.image_normalizer(image=recon_amp_eval, target=target_recon_amp_eval)

            PSNR_layer = utils.psnr2(recon_amp_eval, target_recon_amp_eval).squeeze().tolist()
            if target_recon_amp_eval.size()[1] > 1:  
                # if the tensor has multiple channels, calculate the SSIM for each channel using a self-defined function
                SSIM_layer = self.SSIM_Loss(recon_amp_eval.detach().cpu(), target_recon_amp_eval.detach().cpu()).squeeze().tolist()
            else:
                SSIM_layer = self.SSIM_Loss(recon_amp_eval, target_recon_amp_eval).squeeze().tolist()

            # convert the PSNR and SSIM metrics to the dictionary format
            # average PSNR and SSIM over the first dimension (batch)
            _psnr = np.array(PSNR_layer)
            if _psnr.ndim == 0:
                PSNR_layer = [float(_psnr)]
            elif _psnr.ndim == 2:
                PSNR_layer = _psnr.mean(axis=0).tolist()

            _ssim = np.array(SSIM_layer)
            if _ssim.ndim == 0:
                SSIM_layer = [float(_ssim)]
            elif _ssim.ndim == 2:
                SSIM_layer = _ssim.mean(axis=0).tolist()

            PSNR = {
                'PSNR_layer' : PSNR_layer,
            }

            SSIM = {
                'SSIM_layer' : SSIM_layer,
            }

        return PSNR, SSIM
    
    
    def save_results(self, batch_idx, recon_field, target_recon_field, slm_phase, input_field,
                    epoch, res_dir, PSNR, SSIM, depth_mask=None, zero_shift=False, cali=False, image_id='None'):
        '''Save the results of the neural encoder
        Args:
            batch_idx: the batch index
            recon_field: the reconstruction from phase-only hologram, shape: (batch_size, depth_num, slm_height, slm_width)
            target_recon_field: the target reconstructed from complex hologram, shape: (batch_size, depth_num, slm_height, slm_width)
            slm_phase: the phase-only hologram, shape: (batch_size, 1, slm_height, slm_width)
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
            epoch: the epoch index
            res_dir: the directory for saving the results
            PSNR: the peak signal-to-noise ratio of the reconstructions
            SSIM: the structural similarity index of the reconstructions
            depth_mask: the depth mask stack for 3D image, shape: (batch_size, depth_num, slm_height, slm_width)
            zero_shift: the flag for zero order light shifting
            cali: the flag for calibration
            image_id: the image id, which is extracted from the image file name
        '''
        # create the directory for saving the results
        if os.path.exists(res_dir) and batch_idx == 0:  # remove the existing directory every epoch
            shutil.rmtree(res_dir)
            os.mkdir(res_dir)
        elif not os.path.exists(res_dir):
            os.mkdir(res_dir)

        # create the subdirectories for saving the 3D results
        if self.test_dimension == "3D" and (not cali):
            res_dir = os.path.join(res_dir, str(batch_idx))
            os.mkdir(res_dir)

        # create the subdirectories for saving the different parts of the results
        amp_dir = os.path.join(res_dir, 'amplitude')
        phase_dir = os.path.join(res_dir, 'phase')
        hologram_dir = os.path.join(res_dir, 'hologram')
        complex_dir = os.path.join(res_dir, 'complex hologram')
        target_amp_dir = os.path.join(res_dir, 'target reconstructed amplitude')
        target_phase_dir = os.path.join(res_dir, 'target reconstructed phase')

        if not os.path.exists(hologram_dir):
            os.mkdir(hologram_dir)
        if not os.path.exists(amp_dir):
            os.mkdir(amp_dir)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        if not os.path.exists(complex_dir):
            os.mkdir(complex_dir)
        if not os.path.exists(target_amp_dir):
            os.mkdir(target_amp_dir)
        if not os.path.exists(target_phase_dir):
            os.mkdir(target_phase_dir)

        if not cali:
            if zero_shift:
                linear_phase = torch.arange(slm_phase.size()[2], device=slm_phase.device) * torch.pi   # shift the reconstructed light to the lower part of the zero order light
                slm_phase = slm_phase + linear_phase[None, None, :, None]

            # save the phase-only hologram
            phase_np = slm_phase[0, 0,...].cpu().detach().numpy()
            phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi) # normalize the phase to [0, 1]
            phase_name = "{:s}_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.bmp".format(image_id, epoch, PSNR['PSNR_target'], SSIM['SSIM_target'])
            phase_name = os.path.join(hologram_dir, phase_name)
            cv2.imencode('.bmp', np.uint8(phase_np * 255))[1].tofile(phase_name)

            # save the complex hologram
            field_np = input_field.cpu().detach().numpy()
            field_name = "{:s}_complex_hologram_epoch_{:03d}_No{:03d}.npy".format(image_id, epoch, batch_idx)
            field_name = os.path.join(complex_dir, field_name)
            np.save(field_name, field_np)

            field_phase_np = input_field.real[0, 0,...].cpu().detach().numpy()
            field_phase_np = (field_phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            field_phase_name = "{:s}_field_real_epoch_{:03d}_No{:03d}.bmp".format(image_id, epoch, batch_idx)
            field_phase_name = os.path.join(complex_dir, field_phase_name)
            cv2.imencode('.bmp', np.uint8(field_phase_np * 255))[1].tofile(field_phase_name)

            field_amp_np = input_field.imag[0, 0,...].cpu().detach().numpy()
            field_amp_np = cv2.normalize(field_amp_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            field_amp_name = "{:s}_target_imag_epoch_{:03d}_No{:03d}.bmp".format(image_id, epoch, batch_idx)
            field_amp_name = os.path.join(complex_dir, field_amp_name)
            cv2.imencode('.bmp', np.uint8(field_amp_np * 255))[1].tofile(field_amp_name)

        # center crop the reconstructions and the target reconstructions before saving
        recon_field =  utils.crop_image(recon_field, target_shape=self.roi_res, stacked_complex=False)
        target_recon_field = utils.crop_image(target_recon_field, target_shape=self.roi_res, stacked_complex=False)
        if self.test_dimension == "3D" and (not cali):
            depth_mask = utils.crop_image(depth_mask, target_shape=self.roi_res, stacked_complex=False)

        # save the reconstructions and the target reconstructions for each diffraction distance
        for k in range(self.dist_num if self.test_dimension == "3D" else 1):
            
            # save the amplitude of the reconstructions from phase-only hologram
            I_r_np = recon_field.abs()[0, k, ...]
            I_r_np = I_r_np.cpu().detach().numpy()
            I_r_np = cv2.normalize(I_r_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            I_r_name = "{:s}_layer_{:02d}_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, k, epoch, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
            I_r_name = os.path.join(amp_dir, I_r_name)
            cv2.imencode('.jpg', np.uint8(I_r_np * 255))[1].tofile(I_r_name)

            # save the phase of the reconstructions from phase-only hologram
            phase_np = recon_field.angle()[0, k, ...]
            phase_np = phase_np.cpu().detach().numpy()
            phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            phase_name = "{:s}_layer_{:02d}_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, k, epoch, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
            phase_name = os.path.join(phase_dir, phase_name)
            cv2.imencode('.jpg', np.uint8(phase_np * 255))[1].tofile(phase_name)

            # save the amplitude of the target reconstructions from complex hologram
            target_I_r_np = target_recon_field.abs()[0, k, ...]
            target_I_r_np = target_I_r_np.cpu().detach().numpy()
            target_I_r_np = cv2.normalize(target_I_r_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            target_I_r_name = "{:s}_layer_{:02d}_epoch_{:03d}_No{:03d}.jpg".format(image_id, k, epoch, batch_idx)
            target_I_r_name = os.path.join(target_amp_dir, target_I_r_name)
            cv2.imencode('.jpg', np.uint8(target_I_r_np * 255))[1].tofile(target_I_r_name)

            # save the phase of the target reconstructions from complex hologram
            target_phase_np = target_recon_field.angle()[0, k, ...]
            target_phase_np = target_phase_np.cpu().detach().numpy()
            target_phase_np = (target_phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            target_phase_name = "{:s}_layer_{:02d}_epoch_{:03d}_No{:03d}.jpg".format(image_id, k, epoch, batch_idx)
            target_phase_name = os.path.join(target_phase_dir, target_phase_name)
            cv2.imencode('.jpg', np.uint8(target_phase_np * 255))[1].tofile(target_phase_name)

            # save the phase-only holograms for calibration for each diffraction distance
            # this is only for the experimental calibration
            if cali:
                if zero_shift:
                    linear_phase = torch.arange(slm_phase.size()[2], device=slm_phase.device) * torch.pi   #lower
                    slm_phase  = slm_phase + linear_phase[None, None, :, None]
                phase_np = slm_phase[0, k,...].cpu().detach().numpy()
                phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
                phase_name = "target_phase_layer_{:02d}_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.bmp".format(k, epoch, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
                phase_name = os.path.join(hologram_dir, phase_name)
                cv2.imencode('.bmp', np.uint8(phase_np * 255))[1].tofile(phase_name)

        # save the joint reconstructions for 3D image and depth masks
        if self.test_dimension == "3D" and (not cali):
            joint_dir = os.path.join(res_dir, 'joint')
            mask_dir = os.path.join(res_dir, 'mask')

            if not os.path.exists(joint_dir):
                os.mkdir(joint_dir)
            if not os.path.exists(mask_dir):
                os.mkdir(mask_dir)

            joint_amp = torch.zeros_like(recon_field.abs()[0, 0, ...])
            joint_phase = torch.zeros_like(recon_field.abs()[0, 0, ...])

            # splice the focused parts of the reconstructions at each diffraction distance
            for k in range(self.dist_num):

                joint_amp = joint_amp + recon_field.abs()[0, k, ...] * depth_mask[0, k, ...]
                joint_phase = joint_phase + recon_field.angle()[0, k, ...] * depth_mask[0, k, ...]

                # save the depth masks for each diffraction distance
                mask_np = depth_mask[0, k, ...]
                mask_np = mask_np.cpu().detach().numpy()
                mask_np = cv2.normalize(mask_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
                mask_name = "layer_{:02d}.jpg".format(k)
                mask_name = os.path.join(mask_dir, mask_name)
                cv2.imencode('.jpg', np.uint8(mask_np * 255))[1].tofile(mask_name)

            # save the joint amplitudes of the reconstructions 
            joint_amp_np = joint_amp.cpu().detach().numpy()
            joint_amp_np = cv2.normalize(joint_amp_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            joint_amp_name = "joint_amp_epoch_{:03d}_psnr_{:3f}_ssim_{:4f}.jpg".format(epoch, PSNR['PSNR_target'], SSIM['SSIM_target'])
            joint_amp_name = os.path.join(joint_dir, joint_amp_name)
            cv2.imencode('.jpg', np.uint8(joint_amp_np * 255))[1].tofile(joint_amp_name)

            # save the joint phases of the reconstructions
            joint_phase_np = joint_phase.cpu().detach().numpy()
            joint_phase_np = (joint_phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            joint_phase_name = "joint_phase_epoch_{:03d}.jpg".format(epoch)
            joint_phase_name = os.path.join(joint_dir, joint_phase_name)
            cv2.imencode('.jpg', np.uint8(joint_phase_np * 255))[1].tofile(joint_phase_name)


    def save_results_other_color_3D(self, batch_idx, rgb_recon_field, rgb_target_recon_field, 
                                    rgb_slm_phase, rgb_input_field,
                                    epoch, res_dir, PSNR, SSIM, zero_shift=False, 
                                    cali=False, image_id='None'):
        '''Save the results of the neural encoder
        Args:
            batch_idx: the batch index
            rgb_recon_field: list of the RGB reconstructions from phase-only hologram, list of shape: (batch_size, depth_num, slm_height, slm_width)
            rgb_target_recon_field: list of the RGB targets reconstructed from complex hologram, list of shape: (batch_size, depth_num, slm_height, slm_width)
            rgb_slm_phase: the RGB phase-only holograms, list of shape: (batch_size, 1, slm_height, slm_width)
            rgb_input_field: the RGB complex holograms, list of shape: (batch_size, 1, slm_height, slm_width)
            epoch: the epoch index
            res_dir: the directory for saving the results
            PSNR: the peak signal-to-noise ratio of the reconstructions
            SSIM: the structural similarity index of the reconstructions
            zero_shift: the flag for zero order light shifting
            cali: the flag for calibration
            image_id: the image id, which is extracted from the image file name
        '''
        # create the directory for saving the results
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        # create the subdirectories for saving the 3D results
        if self.test_dimension == "3D" and (not cali):
            res_dir = os.path.join(res_dir, str(image_id))
            if os.path.exists(res_dir) and batch_idx == 0:  # remove the existing directory every epoch
                shutil.rmtree(res_dir)
                os.mkdir(res_dir)
            elif not os.path.exists(res_dir):
                os.mkdir(res_dir)

        # create the subdirectories for saving the different parts of the results
        amp_dir = os.path.join(res_dir, 'amplitude')
        phase_dir = os.path.join(res_dir, 'phase')
        hologram_dir = os.path.join(res_dir, 'hologram')
        complex_dir = os.path.join(res_dir, 'complex hologram')
        target_amp_dir = os.path.join(res_dir, 'target reconstructed amplitude')
        target_phase_dir = os.path.join(res_dir, 'target reconstructed phase')

        if not os.path.exists(hologram_dir):
            os.mkdir(hologram_dir)
        if not os.path.exists(amp_dir):
            os.mkdir(amp_dir)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        if not os.path.exists(complex_dir):
            os.mkdir(complex_dir)
        if not os.path.exists(target_amp_dir):
            os.mkdir(target_amp_dir)
        if not os.path.exists(target_phase_dir):
            os.mkdir(target_phase_dir)

        if not cali:
            # save the phase-only hologram
            for i in reversed(range(3)):
                slm_phase = rgb_slm_phase[i]
                input_field = rgb_input_field[i]
                if zero_shift:
                    linear_phase = torch.arange(slm_phase.size()[2], device=slm_phase.device) * torch.pi   # shift the reconstructed light to the lower part of the zero order light
                    slm_phase = slm_phase + linear_phase[None, None, :, None]
            
                phase_np = slm_phase[0, 0,...].cpu().detach().numpy()
                phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi) # normalize the phase to [0, 1]
                phase_name = "{:s}_channel_{:d}.bmp".format(image_id, i)
                phase_name = os.path.join(hologram_dir, phase_name)
                cv2.imencode('.bmp', np.uint8(phase_np * 255))[1].tofile(phase_name)

                # save the complex hologram
                field_np = input_field.cpu().detach().numpy()
                field_name = "{:s}_complex_hologram_channel_{:d}.npy".format(image_id, i)
                field_name = os.path.join(complex_dir, field_name)
                np.save(field_name, field_np)

                field_phase_np = input_field.real[0, 0,...].cpu().detach().numpy()
                field_phase_np = (field_phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
                field_phase_name = "{:s}_field_real_channel_{:d}.bmp".format(image_id, i)
                field_phase_name = os.path.join(complex_dir, field_phase_name)
                cv2.imencode('.bmp', np.uint8(field_phase_np * 255))[1].tofile(field_phase_name)

                field_amp_np = input_field.imag[0, 0,...].cpu().detach().numpy()
                field_amp_np = cv2.normalize(field_amp_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
                field_amp_name = "{:s}_target_imag_channel_{:d}.bmp".format(image_id, i)
                field_amp_name = os.path.join(complex_dir, field_amp_name)
                cv2.imencode('.bmp', np.uint8(field_amp_np * 255))[1].tofile(field_amp_name)

        # center crop the reconstructions and the target reconstructions before saving
        rgb_recon_field = [utils.crop_image(recon_field, target_shape=self.roi_res, stacked_complex=False) for recon_field in rgb_recon_field]
        rgb_target_recon_field = [utils.crop_image(target_recon_field, target_shape=self.roi_res, stacked_complex=False) for target_recon_field in rgb_target_recon_field]

        # Apply a more robust Gray-World white balance to RGB images across 50 focus depths
        # to make color appearance more consistent across depths.
        eps = 1e-8

        def wb_depth_stack(rgb_list):
            # rgb_list: list of 3 tensors (B, D, H, W); elements can be complex or real.
            B = rgb_list[0].size(0)
            D = rgb_list[0].size(1)
            device = rgb_list[0].device

            # Step 1: For each channel, compute depth-wise scaling factors so each depth mean
            # is normalized toward that channel's global mean across depths.
            per_channel_scaled = []
            for c in range(len(rgb_list)):
                field = rgb_list[c]
                amp = field.abs() if torch.is_complex(field) else field
                # Compute per-depth channel means (across batch and spatial dimensions).
                depth_means = amp.view(B, D, -1).mean(dim=(0, 2))  # (D,)
                # Use the channel's global mean across all depths as the reference.
                global_mean = depth_means.mean()
                scale = (global_mean / (depth_means + eps)).view(1, D, 1, 1).to(device)
                # Clamp the scaling range to avoid excessive amplification or suppression.
                scale = torch.clamp(scale, min=0.5, max=2.0)
                # Scale complex fields directly (preserving phase); real fields are handled similarly.
                per_channel_scaled.append(field * scale)

            # Step 2: Fine-tune at channel level so all three channels have consistent
            # overall brightness (reducing color cast).
            # Compute each channel's mean intensity across all depths, batch, and space.
            channel_means = []
            for field in per_channel_scaled:
                amp = field.abs() if torch.is_complex(field) else field
                # First compute per-depth means, then average over depth to get
                # the channel-level global mean.
                channel_means.append(amp.view(B, D, -1).mean(dim=(0, 2)).mean().item())

            channel_means = torch.tensor(channel_means, device=device, dtype=torch.float32)  # (3,)
            target = channel_means.mean()
            # Compute and apply a gain for each channel (no depth-specific gain here).
            out = []
            for i, field in enumerate(per_channel_scaled):
                gain = (target / (channel_means[i] + eps)).to(device)
                out.append(field * gain)

            return out


        # rgb_recon_field = wb_depth_stack(rgb_recon_field)
        # rgb_target_recon_field = wb_depth_stack(rgb_target_recon_field)
        # recons_np = torch.stack(list(reversed(rgb_recon_field)), dim=-1).detach().cpu().numpy()[0, 25, ...]  # Shape: (H, W, 3)
        # recons_np = (np.abs(recons_np) / np.max(np.abs(recons_np)) * 255).astype(np.uint8)
        # plt.imshow(recons_np)
        # plt.title('Reconstructed Image from Hologram')
        # plt.axis('off')
        # plt.show()
        # save the reconstructions and the target reconstructions for each diffraction distance
        for k in range(self.dist_num if self.test_dimension == "3D" else 1):
            
            # save the amplitude of the reconstructions from phase-only hologram
            I_r_np = [recon_field.abs()[0, k, ...] for recon_field in rgb_recon_field]
            I_r_np_max = max([I_r.max() for I_r in I_r_np])
            I_r_np = [(I_r/I_r_np_max).cpu().detach().numpy() for I_r in I_r_np]
            I_r_np = [I_r.clip(0, 1) for I_r in I_r_np]
            I_r_np = cv2.merge(I_r_np)  # merge the RGB channels
            I_r_name = "{:s}_layer_{:02d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, k, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
            I_r_name = os.path.join(amp_dir, I_r_name)
            cv2.imencode('.jpg', np.uint8(I_r_np * 255))[1].tofile(I_r_name)

            # save the phase of the reconstructions from phase-only hologram
            phase_np = [recon_field.angle()[0, k, ...] for recon_field in rgb_recon_field]
            phase_np = [p.cpu().detach().numpy() for p in phase_np]
            phase_np = [(p + np.pi) % (2 * np.pi) / (2 * np.pi) for p in phase_np]
            phase_np = cv2.merge(phase_np)  # merge the RGB channels
            phase_name = "{:s}_layer_{:02d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, k, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
            phase_name = os.path.join(phase_dir, phase_name)
            cv2.imencode('.jpg', np.uint8(phase_np * 255))[1].tofile(phase_name)

            # save the amplitude of the target reconstructions from complex hologram
            target_I_r_np = [target_recon_field.abs()[0, k, ...] for target_recon_field in rgb_target_recon_field]
            target_I_r_np_max = max([t.max() for t in target_I_r_np])
            target_I_r_np = [(t/target_I_r_np_max).cpu().detach().numpy() for t in target_I_r_np]
            target_I_r_np = [t.clip(0, 1) for t in target_I_r_np]
            target_I_r_np = cv2.merge(target_I_r_np)  # merge the RGB channels
            target_I_r_name = "{:s}_layer_{:02d}.jpg".format(image_id, k)
            target_I_r_name = os.path.join(target_amp_dir, target_I_r_name)
            cv2.imencode('.jpg', np.uint8(target_I_r_np * 255))[1].tofile(target_I_r_name)

            # save the phase of the target reconstructions from complex hologram
            target_phase_np = [target_recon_field.angle()[0, k, ...] for target_recon_field in rgb_target_recon_field]
            target_phase_np = [p.cpu().detach().numpy() for p in target_phase_np]
            target_phase_np = [(p + np.pi) % (2 * np.pi) / (2 * np.pi) for p in target_phase_np]
            target_phase_np = cv2.merge(target_phase_np)  # merge the RGB channels
            target_phase_name = "{:s}_layer_{:02d}.jpg".format(image_id, k)
            target_phase_name = os.path.join(target_phase_dir, target_phase_name)
            cv2.imencode('.jpg', np.uint8(target_phase_np * 255))[1].tofile(target_phase_name)

    

    def save_results_other_color_parallax_3D(self, batch_idx, rgb_recon_parallax_field, rgb_target_parallax_field,
                                    res_dir, PSNR_parallax, SSIM_parallax, zero_shift=False, 
                                    cali=False, image_id='None'):
        '''Save the results of the neural encoder
        Args:
            batch_idx: the batch index
            rgb_recon_parallax_field: list of the RGB reconstructions from phase-only hologram, list of shape: (batch_size, depth_num, slm_height, slm_width)
            rgb_target_parallax_field: list of the RGB targets reconstructed from complex hologram, list of shape: (batch_size, depth_num, slm_height, slm_width)
            res_dir: the directory for saving the results
            PSNR_parallax: the peak signal-to-noise ratio of the reconstructions
            SSIM_parallax: the structural similarity index of the reconstructions
            zero_shift: the flag for zero order light shifting
            cali: the flag for calibration
            image_id: the image id, which is extracted from the image file name
        '''

        res_dir = os.path.join(res_dir, str(image_id))
        # create the subdirectories for saving the different parts of the results
        amp_dir = os.path.join(res_dir, 'parallax amplitude')
        phase_dir = os.path.join(res_dir, 'parallax phase')
        target_amp_dir = os.path.join(res_dir, 'target parallax reconstructed amplitude')
        target_phase_dir = os.path.join(res_dir, 'target parallax reconstructed phase')

        if not os.path.exists(amp_dir):
            os.mkdir(amp_dir)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        if not os.path.exists(target_amp_dir):
            os.mkdir(target_amp_dir)
        if not os.path.exists(target_phase_dir):
            os.mkdir(target_phase_dir)

        # center crop the reconstructions and the target reconstructions before saving
        rgb_recon_parallax_field = [utils.crop_image(recon_field, target_shape=self.roi_res, stacked_complex=False) for recon_field in rgb_recon_parallax_field]
        rgb_target_parallax_field = [utils.crop_image(target_recon_field, target_shape=self.roi_res, stacked_complex=False) for target_recon_field in rgb_target_parallax_field]

        # Apply a more robust Gray-World white balance to RGB images across 50 focus depths
        # to make color appearance more consistent across depths.
        eps = 1e-8

        def wb_depth_stack(rgb_list):
            # rgb_list: list of 3 tensors (B, D, H, W); elements can be complex or real.
            B = rgb_list[0].size(0)
            D = rgb_list[0].size(1)
            device = rgb_list[0].device

            # Step 1: For each channel, compute depth-wise scaling factors so each depth mean
            # is normalized toward that channel's global mean across depths.
            per_channel_scaled = []
            for c in range(len(rgb_list)):
                field = rgb_list[c]
                amp = field.abs() if torch.is_complex(field) else field
                # Compute per-depth channel means (across batch and spatial dimensions).
                depth_means = amp.view(B, D, -1).mean(dim=(0, 2))  # (D,)
                # Use the channel's global mean across all depths as the reference.
                global_mean = depth_means.mean()
                scale = (global_mean / (depth_means + eps)).view(1, D, 1, 1).to(device)
                # Clamp the scaling range to avoid excessive amplification or suppression.
                scale = torch.clamp(scale, min=0.5, max=2.0)
                # Scale complex fields directly (preserving phase); real fields are handled similarly.
                per_channel_scaled.append(field * scale)

            # Step 2: Fine-tune at channel level so all three channels have consistent
            # overall brightness (reducing color cast).
            # Compute each channel's mean intensity across all depths, batch, and space.
            channel_means = []
            for field in per_channel_scaled:
                amp = field.abs() if torch.is_complex(field) else field
                # First compute per-depth means, then average over depth to get
                # the channel-level global mean.
                channel_means.append(amp.view(B, D, -1).mean(dim=(0, 2)).mean().item())

            channel_means = torch.tensor(channel_means, device=device, dtype=torch.float32)  # (3,)
            target = channel_means.mean()
            # Compute and apply a gain for each channel (no depth-specific gain here).
            out = []
            for i, field in enumerate(per_channel_scaled):
                gain = (target / (channel_means[i] + eps)).to(device)
                out.append(field * gain)

            return out

        view_num = rgb_recon_parallax_field[0].size(0)
        # rgb_recon_field = wb_depth_stack(rgb_recon_field)
        # rgb_target_recon_field = wb_depth_stack(rgb_target_recon_field)
        # recons_np = torch.stack(list(reversed(rgb_recon_field)), dim=-1).detach().cpu().numpy()[0, 25, ...]  # Shape: (H, W, 3)
        # recons_np = (np.abs(recons_np) / np.max(np.abs(recons_np)) * 255).astype(np.uint8)
        # plt.imshow(recons_np)
        # plt.title('Reconstructed Image from Hologram')
        # plt.axis('off')
        # plt.show()
        # save the reconstructions and the target reconstructions for each diffraction distance
        for i in range(view_num):
            for j in range(view_num):
            
                # save the amplitude of the reconstructions from phase-only hologram
                I_r_np = [recon_field.abs()[i, j, ...] for recon_field in rgb_recon_parallax_field]
                I_r_np_max = max([I_r.max() for I_r in I_r_np])
                I_r_np = [(I_r/I_r_np_max).cpu().detach().numpy() for I_r in I_r_np]
                I_r_np = [I_r.clip(0, 1) for I_r in I_r_np]
                I_r_np = cv2.merge(I_r_np)  # merge the RGB channels
                I_r_name = "{:s}_view_{:02d}_{:02d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, i, j, PSNR_parallax['PSNR_layer'][i, j], SSIM_parallax['SSIM_layer'][i, j])
                I_r_name = os.path.join(amp_dir, I_r_name)
                cv2.imencode('.jpg', np.uint8(I_r_np * 255))[1].tofile(I_r_name)

                # save the phase of the reconstructions from phase-only hologram
                phase_np = [recon_field.angle()[i, j, ...] for recon_field in rgb_recon_parallax_field]
                phase_np = [p.cpu().detach().numpy() for p in phase_np]
                phase_np = [(p + np.pi) % (2 * np.pi) / (2 * np.pi) for p in phase_np]
                phase_np = cv2.merge(phase_np)  # merge the RGB channels
                phase_name = "{:s}_view_{:02d}_{:02d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, i, j, PSNR_parallax['PSNR_layer'][i, j], SSIM_parallax['SSIM_layer'][i, j])
                phase_name = os.path.join(phase_dir, phase_name)
                cv2.imencode('.jpg', np.uint8(phase_np * 255))[1].tofile(phase_name)

                # save the amplitude of the target reconstructions from complex hologram
                target_I_r_np = [target_recon_field.abs()[i, j, ...] for target_recon_field in rgb_target_parallax_field]
                target_I_r_np_max = max([t.max() for t in target_I_r_np])
                target_I_r_np = [(t/target_I_r_np_max).cpu().detach().numpy() for t in target_I_r_np]
                target_I_r_np = [t.clip(0, 1) for t in target_I_r_np]
                target_I_r_np = cv2.merge(target_I_r_np)  # merge the RGB channels
                target_I_r_name = "{:s}_view_{:02d}_{:02d}.jpg".format(image_id, i, j)
                target_I_r_name = os.path.join(target_amp_dir, target_I_r_name)
                cv2.imencode('.jpg', np.uint8(target_I_r_np * 255))[1].tofile(target_I_r_name)

                # save the phase of the target reconstructions from complex hologram
                target_phase_np = [target_recon_field.angle()[i, j, ...] for target_recon_field in rgb_target_parallax_field]
                target_phase_np = [p.cpu().detach().numpy() for p in target_phase_np]
                target_phase_np = [(p + np.pi) % (2 * np.pi) / (2 * np.pi) for p in target_phase_np]
                target_phase_np = cv2.merge(target_phase_np)  # merge the RGB channels
                target_phase_name = "{:s}_view_{:02d}_{:02d}.jpg".format(image_id, i, j)
                target_phase_name = os.path.join(target_phase_dir, target_phase_name)
                cv2.imencode('.jpg', np.uint8(target_phase_np * 255))[1].tofile(target_phase_name)


    def save_results_other_gray_3D(self, channel, batch_idx, rgb_recon_field, rgb_target_recon_field, 
                                    rgb_slm_phase, rgb_input_field,
                                    epoch, res_dir, PSNR, SSIM, zero_shift=False, 
                                    cali=False, image_id='None'):
        '''Save the results of the neural encoder
        Args:
            batch_idx: the batch index
            rgb_recon_field: list of the grayscale reconstructions from phase-only hologram, list of shape: (batch_size, depth_num, slm_height, slm_width)
            rgb_target_recon_field: list of the RGB targets reconstructed from complex hologram, list of shape: (batch_size, depth_num, slm_height, slm_width)
            rgb_slm_phase: the RGB phase-only holograms, list of shape: (batch_size, 1, slm_height, slm_width)
            rgb_input_field: the RGB complex holograms, list of shape: (batch_size, 1, slm_height, slm_width)
            epoch: the epoch index
            res_dir: the directory for saving the results
            PSNR: the peak signal-to-noise ratio of the reconstructions
            SSIM: the structural similarity index of the reconstructions
            zero_shift: the flag for zero order light shifting
            cali: the flag for calibration
            image_id: the image id, which is extracted from the image file name
        '''
        # create the directory for saving the results
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        # create the subdirectories for saving the 3D results
        if self.test_dimension == "3D" and (not cali):
            res_dir = os.path.join(res_dir, str(image_id))
            if os.path.exists(res_dir) and batch_idx == 0:  # remove the existing directory every epoch
                shutil.rmtree(res_dir)
                os.mkdir(res_dir)
            elif not os.path.exists(res_dir):
                os.mkdir(res_dir)

        # create the subdirectories for saving the different parts of the results
        amp_dir = os.path.join(res_dir, 'amplitude')
        phase_dir = os.path.join(res_dir, 'phase')
        hologram_dir = os.path.join(res_dir, 'hologram')
        complex_dir = os.path.join(res_dir, 'complex hologram')
        target_amp_dir = os.path.join(res_dir, 'target reconstructed amplitude')
        target_phase_dir = os.path.join(res_dir, 'target reconstructed phase')

        if not os.path.exists(hologram_dir):
            os.mkdir(hologram_dir)
        if not os.path.exists(amp_dir):
            os.mkdir(amp_dir)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        if not os.path.exists(complex_dir):
            os.mkdir(complex_dir)
        if not os.path.exists(target_amp_dir):
            os.mkdir(target_amp_dir)
        if not os.path.exists(target_phase_dir):
            os.mkdir(target_phase_dir)

        if not cali:
            # save the phase-only hologram
            slm_phase = rgb_slm_phase[0]
            input_field = rgb_input_field[0]
            if zero_shift:
                linear_phase = torch.arange(slm_phase.size()[2], device=slm_phase.device) * torch.pi   # shift the reconstructed light to the lower part of the zero order light
                slm_phase = slm_phase + linear_phase[None, None, :, None]
        
            phase_np = slm_phase[0, 0,...].cpu().detach().numpy()
            phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi) # normalize the phase to [0, 1]
            phase_name = "{:s}_channel_{:d}.bmp".format(image_id, channel)
            phase_name = os.path.join(hologram_dir, phase_name)
            cv2.imencode('.bmp', np.uint8(phase_np * 255))[1].tofile(phase_name)

            # save the complex hologram
            field_np = input_field.cpu().detach().numpy()
            field_name = "{:s}_complex_hologram_channel_{:d}.npy".format(image_id, channel)
            field_name = os.path.join(complex_dir, field_name)
            np.save(field_name, field_np)

            field_phase_np = input_field.real[0, 0,...].cpu().detach().numpy()
            field_phase_np = (field_phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            field_phase_name = "{:s}_field_real_channel_{:d}.bmp".format(image_id, channel)
            field_phase_name = os.path.join(complex_dir, field_phase_name)
            cv2.imencode('.bmp', np.uint8(field_phase_np * 255))[1].tofile(field_phase_name)

            field_amp_np = input_field.imag[0, 0,...].cpu().detach().numpy()
            field_amp_np = cv2.normalize(field_amp_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            field_amp_name = "{:s}_target_imag_channel_{:d}.bmp".format(image_id, channel)
            field_amp_name = os.path.join(complex_dir, field_amp_name)
            cv2.imencode('.bmp', np.uint8(field_amp_np * 255))[1].tofile(field_amp_name)

        # center crop the reconstructions and the target reconstructions before saving
        rgb_recon_field = utils.crop_image(rgb_recon_field[0], target_shape=self.roi_res, stacked_complex=False)
        rgb_target_recon_field = utils.crop_image(rgb_target_recon_field[0], target_shape=self.roi_res, stacked_complex=False)

        
        for k in range(self.dist_num if self.test_dimension == "3D" else 1):
            
            # save the amplitude of the reconstructions from phase-only hologram
            I_r_np = rgb_recon_field.abs()[0, k, ...]
            I_r_np = I_r_np.cpu().detach().numpy()
            # I_r_np = cv2.normalize(I_r_np.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            I_r_np = I_r_np.clip(0, 1)  # Clip grayscale values directly to [0, 1] to avoid contrast changes from normalization.
            I_r_name = "{:s}_layer_{:02d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, k, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
            I_r_name = os.path.join(amp_dir, I_r_name)
            cv2.imencode('.jpg', np.uint8(I_r_np * 255))[1].tofile(I_r_name)

            # save the phase of the reconstructions from phase-only hologram
            phase_np = rgb_recon_field.angle()[0, k, ...]
            phase_np = (phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            phase_name = "{:s}_layer_{:02d}_psnr_{:3f}_ssim_{:4f}.jpg".format(image_id, k, PSNR['PSNR_layer'][k], SSIM['SSIM_layer'][k])
            phase_name = os.path.join(phase_dir, phase_name)
            cv2.imencode('.jpg', np.uint8(phase_np * 255))[1].tofile(phase_name)

            # save the amplitude of the target reconstructions from complex hologram
            target_I_r_np = rgb_target_recon_field.abs()[0, k, ...]
            target_I_r_np = target_I_r_np.cpu().detach().numpy()
            target_I_r_np = target_I_r_np.clip(0, 1)
            target_I_r_name = "{:s}_layer_{:02d}.jpg".format(image_id, k)
            target_I_r_name = os.path.join(target_amp_dir, target_I_r_name)
            cv2.imencode('.jpg', np.uint8(target_I_r_np * 255))[1].tofile(target_I_r_name)

            # save the phase of the target reconstructions from complex hologram
            target_phase_np = rgb_target_recon_field.angle()[0, k, ...]
            target_phase_np = (target_phase_np + np.pi) % (2 * np.pi) / (2 * np.pi)
            target_phase_name = "{:s}_layer_{:02d}.jpg".format(image_id, k)
            target_phase_name = os.path.join(target_phase_dir, target_phase_name)
            cv2.imencode('.jpg', np.uint8(target_phase_np * 255))[1].tofile(target_phase_name)
