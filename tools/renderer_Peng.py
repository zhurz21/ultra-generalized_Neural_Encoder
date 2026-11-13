import torch
import utils as utils
from .renderer_3D import Renderer
from model import PerceptualLoss_Peng, PerceptualLoss_Vgg11, PerceptualLoss_Vgg16, PerceptualLoss_Vgg13
from tqdm import tqdm, trange
from torch.nn import functional as F


class Renderer_Peng(Renderer):
    '''
    The renderer class for Peng's method, inheriting from the base Renderer class
    '''

    def __init__(self, data, model, device, test_only=False):
        '''Initialize the renderer
        Args:
            data: the meta information of the dataset
            model: the meta information of the HoloNet
            device: the device used for training
            test_only: the flag for testing
        '''
        super(Renderer_Peng, self).__init__(data, model, device, test_only)

        self.vgg_loss = PerceptualLoss_Vgg11(lambda_feat=0.025).to(self.device)  # define the perceptual loss

    
    def layer_prop(self, net, image, cali=False):
        '''Layer-wise propagation for 2D image data
        Args:
            net: the neural encoder
            image: the input image, shape: (batch_size, dist_num, slm_height, slm_width)
            cali: the flag for calibration
        Returns:
            input_field: torch.complex64, the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        '''
        ini_phase = net(input=image, stage='stage_1')  # get the initial phase
        obj_real, obj_imag = utils.polar_to_rect(image, ini_phase)

        object_field = torch.complex(obj_real, obj_imag)


        if self.precomputed_H_inverse.size()[1] > 1:  # for 3D data, the layer-wise propagation is adopted
            interval = 2  # calculate the reconstructions in batches to avoid memory overflow
            input_field = torch.zeros_like(object_field)  # [batch_size, dist_num, slm_height, slm_width]
            for i in range(self.precomputed_H_inverse.size()[1] // interval):

                precomputed_H_inverse = self.precomputed_H_inverse[:, i*interval:(i+1)*interval,...].to(self.device)
                input_field[:, i*interval:(i+1)*interval,...] = self.propagator_backward(u_in=object_field[:, i*interval:(i+1)*interval,...], 
                                                                                         z=-self.prop_dist[:, i*interval:(i+1)*interval],
                                                                                         precomped_H=precomputed_H_inverse)  # gather the reconstructions
        else:    # for 2D data, only one propagation needed
            input_field = self.propagator_backward(u_in=object_field, z=-self.prop_dist, 
                                                    dtype=torch.float32, precomped_H=self.precomputed_H_inverse)  # gather the reconstructions for 2D data

        if not cali:
            input_field = torch.sum(input_field, dim=1, keepdim=True)

        
        return input_field


    def predict(self, net, image, depth_map=None, cali=False):
        '''Predict the phase-only hologram from the input data
        Args:
            net: the neural encoder
            image: the input image, shape: (batch_size, dist_num, slm_height, slm_width)
            depth_map: the depth map, shape: (batch_size, 1, slm_height, slm_width)
            cali: the flag for calibration
        Returns:
            recon_field: the reconstruction from phase-only hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            target_recon_field: the target reconstructed from complex hologram for quantitative evaluation, shape: (batch_size, depth_num, slm_height, slm_width)
            slm_phase: the phase-only hologram, shape: (batch_size, 1, slm_height, slm_width)
            input_field: the complex hologram, shape: (batch_size, 1, slm_height, slm_width)
        '''
        if self.data_type == 'layer':
            input_field = self.layer_prop(net=net, image=image, cali=cali)
        elif self.data_type == 'point':
            raise ValueError("Point Source method is not supported for Peng's method!")

        input_amp = input_field.abs()
        input_field = (input_field - torch.min(input_amp))/(torch.max(input_amp) - torch.min(input_amp))

        input_real = input_field.real
        input_img = input_field.imag
        input_cat = torch.cat([input_real, input_img], 1)  # concatenate the real and imaginary parts of the complex hologram

        input_unshuffle = F.pixel_unshuffle(input_cat, self.downscale_factor)  # shuffle the input data to alleviate the memory consumption

        slm_phase_shuffle = net(input=input_unshuffle, stage='stage_2')
        slm_phase = F.pixel_shuffle(slm_phase_shuffle, self.downscale_factor)  # shuffle the predicted phase-only hologram back to the original resolution
        
        slm_real, slm_imag = utils.polar_to_rect(torch.ones_like(slm_phase), slm_phase)
        slm_field = torch.complex(slm_real, slm_imag)

        recon_field = self.field_reconstructor(slm_field)  # reconstruct the wavefield from phase-only hologram
        target_recon_field = self.field_reconstructor(input_field)  # reconstruct the wavefield from complex hologram


        return recon_field, target_recon_field, slm_phase, input_field
        


    def evaluate(self, 
                 target, 
                 recon_field, 
                 target_recon_field,
                 depth_mask=None,
                 train=True,
                 cali=False):
        ''' Evaluate the performance of the renderer.
        
        Args:
            target: Target image, shape: (batch_size, 1, slm_height, slm_width).
            recon_field: Reconstructed field, shape: (batch_size, depth_num, slm_height, slm_width).
            target_recon_field: Target reconstructed field, shape: (batch_size, depth_num, slm_height, slm_width).
            depth_mask: Depth mask for the input image.
            train: Boolean flag for training mode.
            cali: Boolean flag for calibration mode.
        
        Returns:
            lossValue: Computed loss value.
            PSNR: Peak Signal-to-Noise Ratio metrics.
            SSIM: Structural Similarity Index metrics.
        '''
        lossValue = 0

        recon_amp = recon_field.abs()  # [batch_size, depth_num, slm_height, slm_width]
        target_recon_amp = target_recon_field.abs()
        
        if self.test_dimension == "3D" and (not cali):  # joint the focused parts of the reconstructions at each diffraction distance for 3D image
            joint_recon_amp = torch.sum(recon_amp * depth_mask, dim=1, keepdim=True)
            joint_target_recon_amp = torch.sum(target_recon_amp * depth_mask, dim=1, keepdim=True)

        if cali:
            target = target.repeat(1, recon_amp.size()[1], 1, 1)

        recon_amp_loss = self.image_preprocessor(recon_amp)  # preprocess the reconstructions and the target image
        target_loss = self.image_preprocessor(target)

        if not train:
            recon_amp_loss = self.image_normalizer(image=recon_amp_loss, target=target_loss)  # normalize the reconstructions according to the target image for evaluation

        lossValue = self.vgg_loss(recon_amp_loss.repeat(1, 3, 1, 1), target_loss.repeat(1, 3, 1, 1))  # calculate the loss value with the perceptual loss function
        
        if torch.isnan(lossValue):
            raise ValueError("Zero division error!")
        
        
        with torch.no_grad():

            recon_amp_eval = utils.crop_image(recon_amp, target_shape=self.roi_res, stacked_complex=False)  # crop the ROI region for evaluation
            target_recon_amp_eval =  utils.crop_image(target_recon_amp, target_shape=self.roi_res, stacked_complex=False)
            if self.test_dimension == "3D" and (not cali):
                joint_recon_amp_eval =  utils.crop_image(joint_recon_amp, target_shape=self.roi_res, stacked_complex=False)  # crop the joint image for 3D recontructions
                joint_target_recon_amp_eval =  utils.crop_image(joint_target_recon_amp, target_shape=self.roi_res, stacked_complex=False)
            target_eval = utils.crop_image(target, target_shape=self.roi_res, stacked_complex=False)

            if self.test_dimension == '2D':
                recon_amp_eval = self.image_normalizer(image=recon_amp_eval, target=target_eval)  # normalize the reconstructions
            elif self.test_dimension == '3D':
                recon_amp_eval = self.image_normalizer(image=recon_amp_eval, target=target_recon_amp_eval)

            if self.test_dimension == "3D" and (not cali):
                joint_recon_amp_eval_joint =  self.image_normalizer(image=joint_recon_amp_eval, target=joint_target_recon_amp_eval)
                joint_recon_amp_eval_target =  self.image_normalizer(image=joint_recon_amp_eval, target=target_eval)

            if self.test_dimension == '2D':
                PSNR_layer = utils.psnr2(recon_amp_eval, target_eval).squeeze().tolist()  # calculate the PSNR and SSIM metrics
                SSIM_layer = utils.ssim2(recon_amp_eval, target_eval)
            elif self.test_dimension == '3D':
                PSNR_layer = utils.psnr2(recon_amp_eval, target_recon_amp_eval).squeeze().tolist()
                if target_recon_amp_eval.size()[1] > 1:
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