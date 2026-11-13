import os
from data_operator import Other3DExpSet, CaliSet, get_data_list
from model.builder import build_net
from model.physical_modules import *

from torch.utils.tensorboard import SummaryWriter
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
from tools.renderer_3D import Renderer
from tools.renderer_Peng import Renderer_Peng
import numpy as np
import utils as utils
import time
import datetime
import cv2
import random
import model.loss as myloss
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Pad, InterpolationMode
from model.builder import build_lr_sche, build_net, build_optimizer 
from config import Config
import argparse

# define the unit of the length
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# define the train mode (if you have multiple GPUs, you can use DDP/DP to train the model)
global train_mode
train_mode = None  # "DDP": DistributeDataParallel;
                   # "DP": DataParallel
                   # None: Sigle GPU training

# set the environment variables for DDP
if train_mode == "DDP":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def parse_args():
    # Training settings, you can set the basical training parameters in the command line, 
    # for more detailed settings, you can modify the config file in the config folder (i.e., config/UNet/UNet_config.py)
    parser = argparse.ArgumentParser(description='Train a neural encoder for phase-only hologram') 
    parser.add_argument('--config', help='train config file path', default=os.path.join(os.getcwd(), 'config', 'UNet', 'UNet_LF_parallex_config.py'))
    # parser.add_argument('--config', help='train config file path', default=os.path.join(os.getcwd(), 'config', 'DPM', 'DPM_maimone_LF_parallex_config.py'))
    parser.add_argument('--exp_name', help='name of the experiment', default='UNet_filter_x4_y4_AdamW_bs1_DDP_rectfilter_2w')
    # parser.add_argument('--exp_name', help='name of the experiment', default='AA_DPM_filter_x4_y4_rectfilter')
    # parser.add_argument('--exp_name', help='name of the experiment', default='DPM_maimone_filter_x4_y4_rectfilter')
    # parser.add_argument('--exp_name', help='name of the experiment', default='DDPM_filter_x4_y4_rectfilter')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default=None)
    parser.add_argument('--is_color', help='whether the input data is color', default=True)

    # "layer" means the data is a 2D image or a RGBD 3D data, "point" means the data is a 3D point cloud
    parser.add_argument('--data_type', help='the data format of the data, can be "GS", "mesh", or "LF"', default='LF')
    parser.add_argument('--test_path', help='3D complex hologram dataset used for evaluation', default='Test dataset other3D')

    args = parser.parse_args()

    return args

# set the random seed to ensure the reproducibility of the results
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(rank=None, world_size=None):
    args = parse_args()

    # load the configuration file
    cfg = Config.fromfile(args.config)
    cfg.data.data_type = args.data_type
    cfg.data.test_path = args.test_path

    # choose the device according to the rank
    if train_mode == "DDP":
        if cfg.model.name == 'HoloNet':
            raise ValueError("DDP does not currently support HoloNet.")
        dist.init_process_group("gloo", init_method='env://', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # set the random seed
    setup_seed(cfg.meta_info.Seed)

    # root_dir: the root directory to save the results, checkpoints, logs, and tensorboard
    # if a work_dir is not specified, the default work_dir is set to 'WH/exp_name'
    # else, the work_dir is used
    if args.work_dir == None:
        root_dir = os.path.join(os.getcwd(), 'WH', args.exp_name)
    else:
        root_dir = os.path.join(os.getcwd(), args.work_dir)
    res_dir = os.path.join(root_dir, 'results')
    check_point_dir = os.path.join(root_dir, 'checkpoints')
    log_dir = os.path.join(root_dir, 'logs')
    tensorboard_dir = os.path.join(root_dir, 'tensorboard')

    # put the pathes of the testing data into a list
    test_list = get_data_list(os.path.join(cfg.data.test_path, cfg.data.data_type))

    # define the experiment set
    exp_set = Other3DExpSet(params=cfg.data, data_list=test_list)
    exp_loader = DataLoader(exp_set, batch_size=cfg.meta_info.batch_size_test, shuffle=False)

    # new the folders for saving the results, checkpoints, logs, and tensorboard
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(check_point_dir):
        os.mkdir(check_point_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

        information_line = '='*20 + cfg.data.data_type + '='*20 + '\n'
        log_name = os.path.join(log_dir, 'Log.txt')
        open(log_name, 'a').write(information_line)

    # build the network, the detailed information of the network is defined in the configuration file 
    # (i.e., config/UNet/UNet_config.py)
    net = build_net(cfg.model)

    # set the network to the device
    if train_mode == "DP":
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
        net.to(device)
    if train_mode == "DDP":
        net.to(device)
        net = DDP(net, device_ids=[rank])
    if train_mode == None:
        net.to(device)

    # Load checkpoints
    ckpts_dict = {os.path.splitext(f)[0].split("_")[-1]:os.path.join(check_point_dir, f) 
                  for f in os.listdir(check_point_dir) if f.endswith('pth')}
    # sort the checkpoints according to the global step, ckpts[0] is the best checkpoint, ckpts[-1] is the latest checkpoint
    ckpts_dict = {(0 if dict[0]=="Loss" else 1):dict[1] for dict in ckpts_dict.items()}
    ckpts = [ckpts[1] for ckpts in sorted(ckpts_dict.items())]

    if rank == 0 or rank == None:
        print('Found ckpts', ckpts)
    if len(ckpts) > 0:
        ckpt_path = ckpts[0]  # Load the best checkpoint for evaluation
        if rank == 0 or rank == None:
            print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # update the global step and load the optimizer
        cfg.meta_info.start = ckpt['global_step']+1

        # load the network state dict according to the training mode 
        # (different training mode has different network state dict)
        if train_mode == "DDP":
            # Load model
            net.load_state_dict(ckpt['network_fn_state_dict'])
        else:
            net.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['network_fn_state_dict'].items()})
    

    rgb_recon_field = []
    rgb_target_recon_field = []
    rgb_slm_phase = []
    rgb_input_field = []
    rgb_recon_parallax_field = []
    rgb_target_parallax_field = []
    if args.is_color:
        channel_range = range(3)
    else:
        channel_range = [cfg.meta_info.channel]
    for channel in reversed(channel_range):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cfg.meta_info.channel = channel
        cfg.data.wavelength = cfg.meta_info.wavelength_color[cfg.meta_info.channel]
        cfg.data.channel = cfg.meta_info.channel
        # define the renderer, the renderer is used to predict the hologram and evaluate the results
        # in this code, we provide two renderers, one is the renderer for the Neural encoder\DPM\AADPM\DDPM, 
        # the other is the renderer for the HoloNet 
        # DPM (Maimone er al., Holographic near-eye displays for virtual and augmented reality, 2017)
        # AADPM (Shi et al., Towards real-time photorealistic 3D holography with deep neural networks, 2021)
        # DDPM (Shi et al., End-to-end learning of 3D phase-only holograms for holographic display, 2022)
        # HoloNet (Peng et al., Neural Holography with Camera-in-the-loop Training, 2020)
        if cfg.model.name == 'HoloNet':
            render = Renderer_Peng(data=cfg.data, 
                                model=cfg.model, 
                                device=device, 
                                test_only=True)
        else:
            render = Renderer(data=cfg.data, 
                            model=cfg.model, 
                            device=device, 
                            test_only=True, 
                            model_complex_generator=cfg.model_renderer)

        # evaluate the model on the experimental set
        print(f'evaluation channel {channel}')
        net.eval()   # set the network to the evaluation mode

        psnr = 0.
        ssim = 0.

        # define the folder for saving the evaluation results
        if cfg.meta_info.zero_shift:
            res_dir_eval = os.path.join(res_dir, 'eval_epoch{}_zeroshift'.format(cfg.meta_info.start))
        else:
            res_dir_eval = os.path.join(res_dir, 'eval_epoch{}'.format(cfg.meta_info.start))

        # pop a batch of data from the evaluation set
        # Get the data where batch_idx == channel
        found = False
        for b_idx, (d, image_id, ini_phase) in enumerate(exp_loader):
            if b_idx == channel:
                batch_idx = b_idx
                data = d
                found = True
                break
        if not found:
            raise IndexError(f'Batch with index {channel} not found in exp_loader.')

        image_id = image_id[0]  # get the image id
        data = data.to(device)  # transform the image from CPU to the GPUs
        ini_phase = ini_phase.to(device)

        # during evaluation, the gradient graph is not needed
        with torch.no_grad():
            
            # predict the reconstruction field, the target reconstruction field, the SLM phase, and the input field
            recon_field, target_recon_field, slm_phase, input_field = render.predict_other3D(net, data, ini_phase)
            rgb_recon_field.append(recon_field.detach().cpu())
            rgb_target_recon_field.append(target_recon_field.detach().cpu())
            rgb_slm_phase.append(slm_phase.detach().cpu())
            rgb_input_field.append(input_field.detach().cpu())

            if cfg.data.data_type == "LF":
                recon_parallax_field, target_parallax_field = render.predict_parallax(data, slm_phase, input_field)
                rgb_recon_parallax_field.append(recon_parallax_field.detach().cpu())
                rgb_target_parallax_field.append(target_parallax_field.detach().cpu())

    # calculate the loss, PSNR, and SSIM from the predicted reconstruction field and the target reconstruction field
    PSNR, SSIM = render.evaluate_other3D(torch.concat(rgb_recon_field, dim=0), 
                                         torch.concat(rgb_target_recon_field, dim=0), 
                                         train=False, 
                                         cali=False)

    psnr = psnr + sum(PSNR['PSNR_layer']) / len(PSNR['PSNR_layer']) if isinstance(PSNR['PSNR_layer'], list) else PSNR['PSNR_layer']
    ssim = ssim + sum(SSIM['SSIM_layer']) / len(SSIM['SSIM_layer']) if isinstance(SSIM['SSIM_layer'], list) else SSIM['SSIM_layer']

    if cfg.data.data_type == "LF":
        eval_recon_parallax_field = []
        eval_target_parallax_field = []
        for i in range(len(rgb_recon_parallax_field)):
            num = rgb_recon_parallax_field[i].shape[0]
            eval_recon_parallax_field.append(rgb_recon_parallax_field[i].reshape(1, -1, rgb_recon_parallax_field[i].shape[2], rgb_recon_parallax_field[i].shape[3]))
            eval_target_parallax_field.append(rgb_target_parallax_field[i].reshape(1, -1, rgb_target_parallax_field[i].shape[2], rgb_target_parallax_field[i].shape[3]))

        PSNR_parallax, SSIM_parallax = render.evaluate_other3D(torch.concat(eval_recon_parallax_field, dim=0), 
                                     torch.concat(eval_target_parallax_field, dim=0), 
                                     train=False, 
                                     cali=False)

        PSNR_parallax['PSNR_layer'] = np.array(PSNR_parallax['PSNR_layer']).reshape(num, num)
        SSIM_parallax['SSIM_layer'] = np.array(SSIM_parallax['SSIM_layer']).reshape(num, num)

    # save the evaluation results
    if args.is_color:
        render.save_results_other_color_3D(batch_idx, rgb_recon_field, rgb_target_recon_field, rgb_slm_phase, rgb_input_field,
                            cfg.meta_info.start, res_dir_eval, PSNR, SSIM, 
                            zero_shift=cfg.meta_info.zero_shift, cali=False, image_id=cfg.data.data_type)
        if cfg.data.data_type == "LF":
            render.save_results_other_color_parallax_3D(batch_idx, rgb_recon_parallax_field, rgb_target_parallax_field,
                                                        res_dir_eval, PSNR_parallax, SSIM_parallax,
                                                        zero_shift=cfg.meta_info.zero_shift, cali=False, image_id=cfg.data.data_type)
    else:
        render.save_results_other_gray_3D(channel, batch_idx, rgb_recon_field, rgb_target_recon_field, rgb_slm_phase, rgb_input_field,
                                        cfg.meta_info.start, res_dir_eval, PSNR, SSIM, 
                                        zero_shift=cfg.meta_info.zero_shift, cali=False, image_id=cfg.data.data_type)

    
    psnr = psnr / (1 + batch_idx)
    ssim = ssim / (1 + batch_idx)
    print("Eval psnr: {:2f}; ssim: {:4f}".format(psnr, ssim))


if __name__ == '__main__':
    if train_mode == "DDP":
        world_size= 4  # the number of GPUs
        processes = []
        # spawn the processes for DDP
        for rank in range(world_size):
            p = Process(target=main, args=(rank, world_size))  # define the process
            p.start()  # start the process
            processes.append(p)  # append the process to the list
        for p in processes:
            p.join()
    else:
        main()