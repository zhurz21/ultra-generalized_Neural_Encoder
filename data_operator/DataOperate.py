import os
import glob
import cv2

from torch.utils.data import Dataset
import torch.fft
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Pad, InterpolationMode
import numpy as np
import random
import torch
import utils as utils
import matplotlib.pyplot as plt
from model import propagation_mask


def setup_seed(seed):
     '''set random seed'''

     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_data_list(data_path, ratio=0.8):
    '''
    get the paths of the files in the data_path
    data_path: the parent path needed to be searched
    '''
      
    data_list = glob.glob(os.path.join(data_path, '*'))

    return data_list


def random_mask(slm_res, target_tensor):
    '''
    this function is used to randomly mask the target tensor, 
    this random mask is used to dropout the target tensor,
    which can alleviate the overfitting problem during network training.
    slm_res: the resolution of the SLM, (height, width)
    target_tensor: the target tensor needed to be masked
    '''

    center_h = np.random.randint(0, slm_res[0], size=(1))[0]  # the y coordinate of the center of the mask
    center_w = np.random.randint(0, slm_res[1], size=(1))[0]  # the x coordinate of the center of the mask
    height = np.random.randint(0, slm_res[0]//2, size=(1))[0]  # the height of the mask
    width = np.random.randint(0, slm_res[1]//2, size=(1))[0]  # the width of the mask

    height_up = min(center_h, height)  # the y coordinate of the upper bound of the mask
    height_down = min(slm_res[0]-center_h-1, height)  # the y coordinate of the lower bound of the mask
    width_right = min(center_w, width)  # the x coordinate of the right bound of the mask
    width_left = min(slm_res[1]-center_w-1, width)  # the x coordinate of the left bound of the mask

    target_tensor[:, center_h-height_up:center_h+height_down, center_w-width_right:center_w+width_left] = 0  # mask the target tensor

    return target_tensor



class TrainSet(Dataset):
    '''Define the training dataset'''

    def __init__(self, params):
        '''Initialize the TrainSet'''

        self.using_pdf = params['using_pdf']  # whether to use the probability density function to generate the frequency points
        if self.using_pdf:
            self.conv_h = params['conv_h']  # the standard deviation of the probability density function in the height direction
            self.conv_w = params['conv_w']  # the standard deviation of the probability density function in the width direction
        
        self.slm_res = params['slm_res']  # the resolution of the SLM, (height, width)
        self.roi_res = params['roi_res']  # the resolution of the ROI of the SLM, (height, width)

        self.filter_x = params['filter_x']  # the width of the filter used in Phase Space Tailoring
        self.filter_y = params['filter_y']  # the height of the filter used in Phase Space Tailoring

        self.feature_size = params['feature_size']   # the pixel size of the SLM, (h_pixel, w_pixel)
        # self.wavelength = params['wavelength']  # the wavelength of the light source
        self.prop_dist = params['prop_dist']  # distacne between the SLM and the object

        self.frequency_num_range = params['frequency_num_range']  # the range of the number of frequency points used in 2D Fourier basis amplitude, (min, max)
        self.fraquency_amp_range = params['frequency_amp_range']  # the range of the amplitude of the frequency points used in 2D Fourier basis amplitude, (min, max)
        self.frequency_num_range_phase = params['frequency_num_range_phase']  # the range of the number of frequency points used in 2D Fourier basis phase, (min, max)
        self.fraquency_amp_range_phase = params['frequency_amp_range_phase']  # the range of the amplitude of the frequency points used in 2D Fourier basis phase, (min, max)
        self.ini_phase_type = params['ini_phase_type']  # the type of the initial phase, user can choose from (you can also define their own initial phase): 
                                                        # pseudo_random, pseudo_random_cons, ramdom, 
                                                        # random_cons, zero, FB, FB_cons, FB_constant, 
                                                        # pseudo_random_cons_cons, random_cons_constant
        self.ini_phase_path = os.path.join(os.getcwd(), "ini_phase")  # the path of the pregenerated initial phase

        self.dataset_size = params['data_size']  # the size of the dataset
        self.padding = ((self.slm_res[1]-self.roi_res[1])//2, (self.slm_res[0]-self.roi_res[0])//2)
        self.dropout = params['drop_out']  # the dropout mode of the input data, can be "around" or "embedded"
        if self.dropout == 'in_batch':
            self.dropout_fre = params['drop_out_fre']

        if self.ini_phase_type == 'FB_constant':  # define the Fourier basis phase
            self.ini_phase_file = os.path.join(self.ini_phase_path, 'FB_constant.npy') # the path of the pregenerated initial phase
            if os.path.exists(self.ini_phase_file):  # load the pregenerated initial phase
                self.phase_tensor_constant = torch.from_numpy(np.load(self.ini_phase_file))
            else:  # generate the initial phase if the pregenerated initial phase does not exist
                print('Generate the initial phase for the first time...')
                self.phase_tensor_constant = self.generate_FB_phase(0) * torch.pi - 0.5 * torch.pi  # the phase range is [-0.5*pi, 0.5*pi]
                np.save(self.ini_phase_file, self.phase_tensor_constant.detach().cpu().numpy()) # save the pregenerated initial phase

        elif self.ini_phase_type == 'random_constant':  # define the random phase
            self.ini_phase_file = os.path.join(self.ini_phase_path, 'random_constant.npy') # the path of the pregenerated initial phase
            if os.path.exists(self.ini_phase_file):  # load the pregenerated initial phase
                self.phase_tensor_constant = torch.from_numpy(np.load(self.ini_phase_file))
            else:
                print('Generate the initial phase for the first time...')
                self.phase_tensor_constant = torch.rand(1, *self.slm_res) - 0.5  # the phase range is [-0.5, 0.5]
                np.save(self.ini_phase_file, self.phase_tensor_constant.detach().cpu().numpy()) # save the pregenerated initial phase
    

    def generate_FB_data(self, item):
        '''
        generate the 2D Fourier basis amplitude data,
        item: the index of the dataset, which is used to set the random seed to generate the data randomly
        '''

        setup_seed(item)

        frequency_num = np.random.randint(self.frequency_num_range[0], self.frequency_num_range[1])  # the number of frequency points in a Fourier basis image
        
        if self.using_pdf:
            # generate the frequency points using the probability density function
            mean = [self.slm_res[0]//4, self.slm_res[1]//4]
            cov = np.diag([self.conv_h, self.conv_w])
            index = np.round(np.random.multivariate_normal(mean, cov, frequency_num)).astype(np.int16)  # the sampled frequency points according to the probability density function
            index[index < 1] = 1
            index_h = index[:, 0:1]
            index_w = index[:, 1:2]

            index_h[index_h > (self.slm_res[0]//2 - 1)] = self.slm_res[0]//2 - 1  # the frequency points should be in the range of [0, slm_res//2-1]
            index_w[index_w > (self.slm_res[1]//2 - 1)] = self.slm_res[1]//2 - 1
            index = np.concatenate((index_h, index_w), axis=1)

        else:
            # else, uniformly sample the frequency points within the Phase Space Tailoring filter,
            # this sampling range is used to guarantee the compatibility with phase space tailoring
            index_h = np.random.randint(1, self.slm_res[0]//(self.filter_y//2), size=(frequency_num, 1))
            index_w = np.random.randint(1, self.slm_res[1]//(self.filter_x//2), size=(frequency_num, 1))
        index = np.concatenate((index_h, index_w), axis=1)
        fraquency_amp = np.random.random((frequency_num)) * (self.fraquency_amp_range[1] - self.fraquency_amp_range[0])  # define the amplitude of the frequency points
        fraquency_amp = fraquency_amp + self.fraquency_amp_range[0]

        FB_mask = torch.zeros(1, 1, self.slm_res[0]//(self.filter_y//2), self.slm_res[1]//(self.filter_x//2), dtype=torch.float32)  # [1, 1, PST_height, PST_width]

        for i in range(frequency_num):
            # to realize the Hermitian symmetry in the Fourier domain,
            # the amplitude of the frequency points are assigned to the corresponding symmetric points (slm_res[0]//2-index_h, slm_res[1]//2-index_w)
            # the center point of the PST filter (slm_res[0]//filter_y, slm_res[1]//filter_x) is zero-shifted
            # which can produce a real-valued image in the spatial domain
            FB_mask[..., index[i, 0], index[i, 1]] = fraquency_amp[i]
            FB_mask[..., self.slm_res[0]//(self.filter_y//2)-index[i, 0], self.slm_res[1]//(self.filter_x//2)-index[i, 1]] = fraquency_amp[i]
            FB_mask[..., self.slm_res[0]//self.filter_y, self.slm_res[1]//self.filter_x] += fraquency_amp[i]*2
        FB_mask[..., self.slm_res[0]//self.filter_y, self.slm_res[1]//self.filter_x] += 0.1


        imaug_mask = Compose([
                        Pad((self.slm_res[1]//self.filter_x, self.slm_res[0]//self.filter_y), fill=0, padding_mode='constant'),
                        ])    # padding the mask to the resolution of the SLM

        FB_mask = imaug_mask(FB_mask)

        target_field = propagation_mask(mask=FB_mask).squeeze(0)   # Fourier transform the mask to the spatial domain

        target_amp = target_field.abs()  # the amplitude of the Fourier tansformed image is the Fourier basis image
        target_phase = target_field.angle()

        # normalize the Fourier basis image to [0, 1]
        if torch.max(target_amp) - torch.min(target_amp) != 0:
            target_amp = (target_amp - torch.min(target_amp))/(torch.max(target_amp) - torch.min(target_amp))
        else:
            target_amp = target_amp/torch.min(target_amp)

        # dropout the Fourier basis image according to the selected dropout mode
        if self.dropout == 'around':    # dropout the image around the center of the image
            imaug = Compose([
                            CenterCrop(self.roi_res),
                            Pad(self.padding, fill=0, padding_mode='constant'),
                            ])

            target_tensor = imaug(target_amp)
        elif self.dropout == 'embedded':   # dropout the image with random mask
            target_tensor = random_mask(self.slm_res, target_amp)

        return target_tensor
    

    def generate_FB_phase(self, item):
        '''
        This function is used to generate the Fourier basis phase,
        which is exactly the same as the function "generate_FB_data()", 
        except for the sampling parameters of the Fourier points.
        '''

        setup_seed(item)

        frequency_num = np.random.randint(self.frequency_num_range_phase[0], self.frequency_num_range_phase[1])
        
        if self.using_pdf:
            mean = [self.slm_res[0]//4, self.slm_res[1]//4]
            cov = np.diag([self.conv_h, self.conv_w])
            index = np.round(np.random.multivariate_normal(mean, cov, frequency_num)).astype(np.int16)
            index[index < 1] = 1
            index_h = index[:, 0:1]
            index_w = index[:, 1:2]
            index_h[index_h > (self.slm_res[0]//2 - 1)] = self.slm_res[0]//2 - 1
            index_w[index_w > (self.slm_res[1]//2 - 1)] = self.slm_res[1]//2 - 1
            index = np.concatenate((index_h, index_w), axis=1)

        else:
            index_h = np.random.randint(1, self.slm_res[0]//(self.filter_y//2), size=(frequency_num, 1))
            index_w = np.random.randint(1, self.slm_res[1]//(self.filter_x//2), size=(frequency_num, 1))
        index = np.concatenate((index_h, index_w), axis=1)
        fraquency_amp = np.random.random((frequency_num)) * (self.fraquency_amp_range_phase[1] - self.fraquency_amp_range_phase[0])
        fraquency_amp = fraquency_amp + self.fraquency_amp_range_phase[0]

        FB_mask = torch.zeros(1, 1, self.slm_res[0]//(self.filter_y//2), self.slm_res[1]//(self.filter_x//2), dtype=torch.float32)

        for i in range(frequency_num):
            FB_mask[..., index[i, 0], index[i, 1]] = fraquency_amp[i]
            FB_mask[..., self.slm_res[0]//(self.filter_y//2)-index[i, 0], self.slm_res[1]//(self.filter_x//2)-index[i, 1]] = fraquency_amp[i]
            FB_mask[..., self.slm_res[0]//self.filter_y, self.slm_res[1]//self.filter_x] += fraquency_amp[i]*2
        FB_mask[..., self.slm_res[0]//self.filter_y, self.slm_res[1]//self.filter_x] += 0.1


        imaug_mask = Compose([
                        Pad((self.slm_res[1]//self.filter_x, self.slm_res[0]//self.filter_y), fill=0, padding_mode='constant'),
                        ])

        FB_mask = imaug_mask(FB_mask)

        target_field = propagation_mask(mask=FB_mask).squeeze(0)

        target_amp = target_field.abs()
        target_phase = target_field.angle()

        if torch.max(target_amp) - torch.min(target_amp) != 0:
            target_amp = (target_amp - torch.min(target_amp))/(torch.max(target_amp) - torch.min(target_amp))
        else:
            target_amp = target_amp/torch.min(target_amp)

        if self.dropout == 'around':
            imaug = Compose([
                            CenterCrop(self.roi_res),
                            Pad(self.padding, fill=0, padding_mode='constant'),
                            ])

            target_tensor = imaug(target_amp)
        elif self.dropout == 'embedded':
            target_tensor = random_mask(self.slm_res, target_amp)

        return target_tensor


    def __getitem__(self, item):
        '''Get the item of the dataset'''
        
        target_tensor = self.generate_FB_data(item)  # generate the Fourier basis amplitude data

        # generate the initial phase data according to the selected initial phase type
        if self.ini_phase_type == 'FB':
            phase_tensor = self.generate_FB_phase(item) * 2 * torch.pi - torch.pi
        elif self.ini_phase_type == 'FB_cons':
            phase_tensor = self.generate_FB_phase(item) * torch.pi - 0.5 * torch.pi
        elif self.ini_phase_type == 'FB_constant':
            phase_tensor = self.phase_tensor_constant
        elif self.ini_phase_type == 'random_cons_constant':
            phase_tensor = self.phase_tensor_constant
        elif self.ini_phase_type == 'pseudo_random_cons_cons':
            ini_phase_name = os.path.join(self.ini_phase_path, 'pseudo_random_cons.npy')  # the pseudo random phase is pergenerated
            phase_tensor = torch.from_numpy(np.load(ini_phase_name)).squeeze(0)
            phase_tensor = phase_tensor * 0.5
        elif self.ini_phase_type == 'zero':
            phase_tensor = torch.zeros_like(target_tensor)
        else:
            ini_phase_name = os.path.join(self.ini_phase_path, self.ini_phase_type+'.npy')  # you can also define your own initial phase
            phase_tensor = torch.from_numpy(np.load(ini_phase_name)).squeeze(0)

        return target_tensor, target_tensor, phase_tensor


    def __len__(self):
        return self.dataset_size



class TestSet(TrainSet):
    '''Define the test dataset'''
    
    def __init__(self, params, data_list):

        super().__init__(params) # inherit the initialization of the TrainSet
        
        self.channel = params['channel']
        self.slm_res = params['slm_res']
        self.roi_res = params['roi_res']
        self.test_dimension = params['test_dimension']  # the dimension of the data used for evaluation, can be "2D" or "3D"
        self.padding = ((self.slm_res[1]-self.roi_res[1])//2, (self.slm_res[0]-self.roi_res[0])//2)
        self.un_zero_blk = params['un_zero_blk']  # zero-shift the black regions of the input data with a samll value
        self.ini_phase_type = params['ini_phase_type']  # the initial phase used in testing

        self.data_list = data_list  # the paths of the test images

        if self.test_dimension == '3D':
            self.depth_list = [file.replace('img', 'depth') for file in self.data_list]   # the paths of the depth images


    def __getitem__(self, item):

        data_name = self.data_list[item]
        data_tensor = self.augment_img_reader(data_name)
        image_id, image_extension = os.path.splitext(os.path.basename(data_name))  # get the image id from the image name

        if self.test_dimension == '3D':
            depth_name = self.depth_list[item]
            depth_tensor = self.augment_img_reader(depth_name, cal_depth=True)
            image_tensor = torch.cat([data_tensor, depth_tensor], dim=0)
        else:
            image_tensor = data_tensor

        # generate the initial phase data according to the selected initial phase type
        if self.ini_phase_type == 'FB':
            phase_tensor = self.generate_FB_phase(item) * 2 * torch.pi - torch.pi
        elif self.ini_phase_type == 'FB_cons':
            phase_tensor = self.generate_FB_phase(item) * torch.pi - 0.5 * torch.pi
        elif self.ini_phase_type == 'FB_constant':
            phase_tensor = self.phase_tensor_constant
        elif self.ini_phase_type == 'random_cons_constant':
            phase_tensor = self.phase_tensor_constant
        elif self.ini_phase_type == 'pseudo_random_cons_cons':
            ini_phase_name = os.path.join(self.ini_phase_path, 'pseudo_random_cons.npy')
            phase_tensor = torch.from_numpy(np.load(ini_phase_name)).squeeze(0)
            phase_tensor = phase_tensor * 0.5
        elif self.ini_phase_type == 'zero':
            phase_tensor = torch.zeros((1, *image_tensor.size()[-2:]))
        else:
            ini_phase_name = os.path.join(self.ini_phase_path, self.ini_phase_type+'.npy')
            phase_tensor = torch.from_numpy(np.load(ini_phase_name)).squeeze(0)
        
        return image_tensor, data_tensor, image_id, phase_tensor



    def augment_img_reader(self, data_name, cal_depth=False):
        '''Read the image data and augment the image'''

        data = cv2.imdecode(np.fromfile(data_name, dtype=np.uint8),cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = data[..., self.channel]  # extract the channel needed
        data = cv2.normalize(data.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        
        data = data[np.newaxis, :, :]
        data = torch.from_numpy(data).type(torch.float32)  # convert the data to tensor

        image_resize = Resize(self.roi_res[0])  # resize the image according to the height of the ROI
        data = image_resize(data)

        if data.size()[-1] >= self.roi_res[1]:  # crop the image according to the width of the ROI
            image_crop = CenterCrop(self.roi_res)
            data = image_crop(data)
        else:  # pad the image if the width of the image is smaller than the width of the ROI
            image_pad = Pad(((self.roi_res[1]-data.size()[-1])//2, 0), fill=0, padding_mode='constant')
            data = image_pad(data)

        imaug = Compose([
                        Resize(self.roi_res),
                        Pad(self.padding, fill=0, padding_mode='constant'),
                        ])


        data_tensor = imaug(data)

        if self.un_zero_blk:
            if not cal_depth:
                data_tensor[data_tensor == 0] = 0.05  # zero-shift the black regions of the input data with a samll value

        # plt.imshow(data_tensor[0,...], cmap='gray')
        # plt.show()

        return data_tensor



    def __len__(self):
        return len(self.data_list)


class ExpSet(TestSet):
    '''Define the experiment dataset'''
    
    def augment_img_reader(self, data_name, cal_depth=False):
        '''Read the image data and augment the image'''

        data = cv2.imdecode(np.fromfile(data_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = data[..., self.channel]  # extract the channel needed
        data = cv2.normalize(data.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        
        data = data[np.newaxis, :, :]
        data = torch.from_numpy(data).type(torch.float32)

        image_resize_h = Resize((self.roi_res[0], int(data.size()[-1]*self.roi_res[0]/data.size()[-2])))
        data_h = image_resize_h(data)  # resize the image according to the height of the ROI

        image_resize_w = Resize((int(data.size()[-2]*self.roi_res[1]/data.size()[-1]), self.roi_res[1]))
        data_w = image_resize_w(data)  # resize the image according to the width of the ROI

        # make sure that the image content can fill the ROI
        if data_h.size()[-1] >= self.roi_res[1]:
            image_crop = CenterCrop(self.roi_res)
            data_tensor = image_crop(data_h)

        if data_w.size()[-2] >= self.roi_res[0]:
            image_crop = CenterCrop(self.roi_res)
            data_tensor = image_crop(data_w)

        if self.test_dimension == "3D":   # if the data is 3D, pad the image with reflected image content
            imaug = Compose([
                        Resize(self.roi_res),
                        Pad(self.padding, padding_mode='reflect'),
                        ]) 
        else:  # if the data is 2D, pad the image with constant value
            imaug = Compose([
                            Resize(self.roi_res),
                            Pad(self.padding, fill=0, padding_mode='constant'),
                            ])

        data_tensor = imaug(data_tensor)

        if self.un_zero_blk:
            if not cal_depth:
                data_tensor[data_tensor == 0] = 0.05  # zero-shift the black regions of the input data with a samll value


        return data_tensor



class Other3DExpSet(TrainSet):
    '''Define the experimental dataset for other 3D data like Gaussian splatting, mesh and light field data'''
    
    def __init__(self, params, data_list):

        super().__init__(params) # inherit the initialization of the TrainSet
        
        self.slm_res = params['slm_res']
        self.roi_res = params['roi_res']
        self.test_dimension = params['test_dimension']  # the dimension of the data used for evaluation, can be "2D" or "3D"
        self.padding = ((self.slm_res[1]-self.roi_res[1])//2, (self.slm_res[0]-self.roi_res[0])//2)
        self.ini_phase_type = params['ini_phase_type']  # the initial phase used in testing

        self.data_list = data_list  # the paths of the test images


    def __getitem__(self, item):

        data_name = self.data_list[item]
        data_tensor = torch.from_numpy(np.load(data_name)).to(torch.complex64).unsqueeze(0)  # load the complex field data
        imaug = Compose([
                        Pad(((self.slm_res[1]-data_tensor.shape[-1])//2, (self.slm_res[0]-data_tensor.shape[-2])//2), fill=0, padding_mode='constant'),
                        ])
        data_tensor = imaug(data_tensor)
        image_id, image_extension = os.path.splitext(os.path.basename(data_name))  # get the image id from the image name

        # generate the initial phase data according to the selected initial phase type
        if self.ini_phase_type == 'FB':
            phase_tensor = self.generate_FB_phase(item) * 2 * torch.pi - torch.pi
        elif self.ini_phase_type == 'FB_cons':
            phase_tensor = self.generate_FB_phase(item) * torch.pi - 0.5 * torch.pi
        elif self.ini_phase_type == 'FB_constant':
            phase_tensor = self.phase_tensor_constant
        elif self.ini_phase_type == 'random_cons_constant':
            phase_tensor = self.phase_tensor_constant
        elif self.ini_phase_type == 'pseudo_random_cons_cons':
            ini_phase_name = os.path.join(self.ini_phase_path, 'pseudo_random_cons.npy')
            phase_tensor = torch.from_numpy(np.load(ini_phase_name)).squeeze(0)
            phase_tensor = phase_tensor * 0.5
        elif self.ini_phase_type == 'zero':
            phase_tensor = torch.zeros((1, *data_tensor.size()[-2:]))
        else:
            ini_phase_name = os.path.join(self.ini_phase_path, self.ini_phase_type+'.npy')
            phase_tensor = torch.from_numpy(np.load(ini_phase_name)).squeeze(0)
        
        return data_tensor, image_id, phase_tensor


    def __len__(self):
        return len(self.data_list)



class CaliSet(Dataset):

    '''
    Generate calibration image for the calibration of the holograhic reconstruction during experiment

    initialization:
    slm_res: the resolution of the SLM
    roi_res: the resolution of the ROI of the SLM
    num_circles: the number of circle points in the calibration image
    circle_radius: the radius of the circle point in the calibration image

    returns:
    cali_image: the calibration image
    '''
    
    def __init__(self, data, cali):
        
        self.slm_res = data['slm_res']
        self.roi_res = data['roi_res']

        self.num_circles = cali['num_circles']
        self.circle_radius = cali['circle_radius']


    def __getitem__(self, item):
        
        # generate calibration image using the function in utils
        cali_image = utils.generate_calibration_img(self.slm_res, self.roi_res, self.num_circles, self.circle_radius)

        cali_image = cv2.normalize(cali_image.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        
        cali_image = cali_image[np.newaxis, :, :]
        cali_image = torch.from_numpy(cali_image).type(torch.float32)

        return cali_image, cali_image

    def __len__(self):
        return 1