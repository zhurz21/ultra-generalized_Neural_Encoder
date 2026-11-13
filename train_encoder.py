import os
from data_operator import TrainSet, TestSet, ExpSet, CaliSet, get_data_list
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
    parser.add_argument('--config', help='train config file path', default=os.path.join(os.getcwd(), 'config', 'UNet', 'UNet_config.py'))
    parser.add_argument('--exp_name', help='name of the experiment', default='test')
    parser.add_argument('--test_only', help='run the code in evaluation mode', action='store_true')
    parser.add_argument('--no_reload', help='do not train from the checkpoint', action='store_true')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default=None)

    parser.add_argument('--test_dimension', help='the dimension of the data used for evaluation, can be "2D" or "3D"', default="2D")
    # "layer" means the data is a 2D image or a RGBD 3D data, "point" means the data is a 3D point cloud
    parser.add_argument('--data_type', help='the data format of the data, can be "layer", "point", or "end-to-end"', default='layer') 
    parser.add_argument('--test_path', help='2D or 3D image dataset used for evaluation', default='Test dataset')

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
    cfg.data.test_dimension = args.test_dimension
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
    if cfg.data.test_dimension == "2D":
        test_list = get_data_list(cfg.data.test_path)
    else:
        test_list = get_data_list(os.path.join(cfg.data.test_path, 'img'))

    # define the training set, testing set, and the experiment set
    # as the training set is generated randomly during each iteration, the training set is not saved in the disk
    train_set = TrainSet(params=cfg.data)
    test_set = TestSet(params=cfg.data, data_list=test_list)
    exp_set = ExpSet(params=cfg.data, data_list=test_list)

    if train_mode == "DDP":
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = DataLoader(train_set, batch_size=cfg.meta_info.batch_size_train, num_workers=4,
                                               pin_memory=True, sampler=train_sampler)

    else:
        train_loader = DataLoader(train_set, batch_size=cfg.meta_info.batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.meta_info.batch_size_test, shuffle=False)
    exp_loader = DataLoader(exp_set, batch_size=cfg.meta_info.batch_size_test, shuffle=False)

    # new the folders for saving the results, checkpoints, logs, and tensorboard
    if args.test_only and (not os.path.exists(res_dir)):
        os.mkdir(res_dir)
    if not args.test_only:
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

        information_line = '='*20 + ' FB ' + '='*20 + '\n'
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

    # build the optimizer and the learning rate scheduler, 
    # the detailed information of the optimizer and the learning rate scheduler is defined in the configuration file
    # (i.e., config/UNet/UNet_config.py)
    optvars = [{'params': net.parameters()}]
    optimizer = build_optimizer(cfg.optimizer, model_params=optvars)
    scheduler = build_lr_sche(cfg.optimizer.lr_scheduler, optimizer=optimizer)

    # tensorboard
    if not args.test_only:
        filename = 'FB'
        writer = SummaryWriter(os.path.join(tensorboard_dir, 'tensorboard_' + filename))
        writer_freq = 10

    # Load checkpoints
    ckpts_dict = {os.path.splitext(f)[0].split("_")[-1]:os.path.join(check_point_dir, f) 
                  for f in os.listdir(check_point_dir) if f.endswith('pth')}
    # sort the checkpoints according to the global step, ckpts[0] is the best checkpoint, ckpts[-1] is the latest checkpoint
    ckpts_dict = {(0 if dict[0]=="Loss" else 1):dict[1] for dict in ckpts_dict.items()}
    ckpts = [ckpts[1] for ckpts in sorted(ckpts_dict.items())]

    if rank == 0 or rank == None:
        print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        if args.test_only:
            ckpt_path = ckpts[0]  # Load the best checkpoint for evaluation
        else:
            ckpt_path = ckpts[-1] # Load the latest checkpoint for training
        if rank == 0 or rank == None:
            print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # update the global step and load the optimizer
        cfg.meta_info.start = ckpt['global_step']+1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # load the network state dict according to the training mode 
        # (different training mode has different network state dict)
        if train_mode == "DDP":
            # Load model
            net.load_state_dict(ckpt['network_fn_state_dict'])
        else:
            net.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['network_fn_state_dict'].items()})

    # define the renderer, the renderer is used to predict the hologram and evaluate the results
    # in this code, we provide two renderers, one is the renderer for the Neural encoder\DPM\AADPM\DDPM, 
    # the other is the renderer for the HoloNet 
    # DPM (Maimone er al., Holographic near-eye displays for virtual and augmented reality, 2017)
    # AADPM (Shi et al., Towards real-time photorealistic 3D holography with deep neural networks, 2021)
    # DDPM (Shi et al., End-to-end learning of 3D phase-only holograms for holographic display, 2022)
    # HoloNet (Peng et al., Neural Holography with Camera-in-the-loop Training, 2020)
    if cfg.model.name == 'HoloNet':
        render = Renderer_Peng(cfg.data, cfg.model, device, args.test_only, cfg.model_renderer)
    else:
        render = Renderer(cfg.data, cfg.model, device, args.test_only, cfg.model_renderer)

    # generate the holograms of the lattice diagram for the calibration during experiment
    if cfg.meta_info.calibration:
        cali_set = CaliSet(cfg.data, cfg.cali)
        cali_loader = DataLoader(cali_set, batch_size=1, shuffle=False)

        print('calibration generating...')
        net.eval()  # set the network to the evaluation mode

        loss = 0.
        psnr = 0.
        ssim = 0.

        # define the folder for saving the calibration results
        if cfg.meta_info.zero_shift:
            res_dir_eval = os.path.join(res_dir, 'cali_epoch{}_{}x{}_{}nm_zeroshift'.format(cfg.meta_info.start, cfg.cali['num_circles'][0], 
                                                                                            cfg.cali['num_circles'][1], cfg.data.wavelength/nm))
        else:
            res_dir_eval = os.path.join(res_dir, 'cali_epoch{}_{}x{}_{}nm'.format(cfg.meta_info.start, cfg.cali['num_circles'][0], 
                                                                                  cfg.cali['num_circles'][1], cfg.data.wavelength/nm))

        # pop the lattice diagram for calibration
        for batch_idx, (image, target) in enumerate(cali_loader):
            image = image.to(device)
            target = target.to(device)

            with torch.no_grad():
                # encoding the phase-only hologram according to the lattice diagram
                recon_field, target_recon_field, slm_phase, input_field = render.predict(net, 
                                                                                         image, 
                                                                                         ini_phase=torch.zeros_like(image), 
                                                                                         cali=True)

                # calculate the loss, PSNR, and SSIM from the predicted reconstruction field and the target reconstruction field
                lossValue, PSNR, SSIM = render.evaluate(target, recon_field, target_recon_field, train=False, cali=True)
                loss = loss + lossValue
                psnr = psnr + sum(PSNR['PSNR_layer']) / len(PSNR['PSNR_layer']) if isinstance(PSNR['PSNR_layer'], list) else PSNR['PSNR_layer']
                ssim = ssim + sum(SSIM['SSIM_layer']) / len(SSIM['SSIM_layer']) if isinstance(SSIM['SSIM_layer'], list) else SSIM['SSIM_layer']

                # save the calibration results, including the predicted reconstruction field, 
                # the target reconstruction field, the SLM phase, and the input field
                render.save_results(batch_idx, recon_field, target_recon_field, slm_phase, input_field,
                                    cfg.meta_info.start, res_dir_eval, PSNR, SSIM, zero_shift=cfg.meta_info.zero_shift, cali=True)

        
        loss = loss / (1 + batch_idx)
        psnr = psnr / (1 + batch_idx)
        ssim = ssim / (1 + batch_idx)
        print("Calibration loss: {}; psnr: {:2f}; ssim: {:4f}".format(loss, psnr, ssim))


    # training the network
    if not args.test_only:
        for epoch in range(cfg.meta_info.start, cfg.meta_info.epoch_number):

            epoch_start_time = time.time() # record the start time of the epoch
            if rank == 0 or rank == None:
                print("Epoch: {}".format(epoch)) 
            epoch_loss = 0.
            net.train()
            start_time = time.time()
            loss = 0.
            psnr = 0.
            ssim = 0.

            for batch_idx, (image, target, ini_phase) in enumerate(train_loader):
                torch.cuda.empty_cache()  # empty the cache of the GPU

                image = image.to(device)  # put the image from CPU to the GPUs
                target = target.to(device)
                ini_phase = ini_phase.to(device)

                optimizer.zero_grad()

                # predict the reconstruction field, the target reconstruction field, the SLM phase, and the input field
                recon_field, target_recon_field, slm_phase, input_field = render.predict(net, image, ini_phase)

                # calculate the loss, PSNR, and SSIM from the predicted reconstruction field and the target reconstruction field
                lossValue, PSNR, SSIM = render.evaluate(target, recon_field, target_recon_field, train=True, cali=False)

                # record the loss, PSNR, and SSIM
                epoch_loss += lossValue.item()
                psnr = psnr + sum(PSNR['PSNR_layer']) / len(PSNR['PSNR_layer']) if isinstance(PSNR['PSNR_layer'], list) else PSNR['PSNR_layer']
                ssim = ssim + sum(SSIM['SSIM_layer']) / len(SSIM['SSIM_layer']) if isinstance(SSIM['SSIM_layer'], list) else SSIM['SSIM_layer']

                lossValue.backward()  # backpropagation
                optimizer.step() # update the parameters of the network

                if rank == 0 or rank == None:
                    if batch_idx % 10 == 0:    # print the information every 10 batches
                        batch_time = time.time() - start_time  # calculate the time cost of the 10 batches
                        remain_time = utils.format_time((len(train_loader) - batch_idx) * batch_time/10)  # calculate the remaining time
                        print_line = 'Epoch: {} | Batch: {} -----> Lr: {:4f}e-3, Train loss: {:4f}, psnr: {:2f}, ssim: {:4f}, Cost Time: {}, Remain Time: {}' \
                                    '' \
                            .format(epoch, batch_idx, 
                                    optimizer.state_dict()['param_groups'][0]['lr']*1e3, 
                                    epoch_loss / (batch_idx + 1),
                                    psnr/(batch_idx + 1), 
                                    ssim/(batch_idx + 1), 
                                    batch_time, remain_time)
                        print(print_line)
                        start_time = time.time()

            # save and print the information of the epoch
            if rank == 0 or rank == None:  
    
                if scheduler != None:
                    scheduler.step()  # update the learning rate

                lr_now = optimizer.state_dict()['param_groups'][0]['lr']

                loss = epoch_loss / (1 + batch_idx)
                psnr = psnr / (1 + batch_idx)
                ssim = ssim / (1 + batch_idx)

                print('Epoch {} Finished ! Loss is {:4f}'.format(epoch, loss))
                open(log_name, 'a') \
                    .write("Epoch {} Lr: {}e-3,  Loss: {}; psnr: {:2f}; ssim: {:4f} \n".format(epoch, 
                                                                                            lr_now*1e3, 
                                                                                            loss,
                                                                                            psnr, 
                                                                                            ssim))

                print("Epoch time: ", time.time() - epoch_start_time)

                # viasualize the results in the tensorboard
                utils.to_tensorboard(epoch,
                                     recon_field.abs(),
                                     recon_field.angle(),
                                     target,
                                     lr_now,
                                     loss,
                                     psnr,
                                     ssim,
                                     writer,
                                     'train')

                # validate the network
                if (epoch % cfg.meta_info.epoch_save_fre) == 0 and (epoch != cfg.meta_info.epoch_number-1):

                    net.eval()

                    loss = 0.
                    psnr = 0.
                    ssim = 0.
                    
                    # pop a batch of data from the validation set
                    for batch_idx, (image, target, image_id, ini_phase) in enumerate(test_loader):
                        
                        image_id = image_id[0]  # get the image id
                        image = image.to(device)  # transform the image from CPU to the GPUs
                        target = target.to(device)
                        ini_phase = ini_phase.to(device)

                        with torch.no_grad():
                            # the same as the training process, the difference is that the network is set to the evaluation mode
                            recon_field, target_recon_field, slm_phase, input_field = render.predict(net, image, ini_phase)

                            # calculate the loss, PSNR, and SSIM from the predicted reconstruction field and the target reconstruction field
                            lossValue, PSNR, SSIM = render.evaluate(target, recon_field, target_recon_field, train=False, cali=False)

                            # define the folder for saving the validate results
                            res_dir_epoch = os.path.join(res_dir, 'validate')

                            # save the validate results
                            render.save_results(batch_idx, recon_field, target_recon_field, slm_phase, input_field,
                                                epoch, res_dir_epoch, PSNR, SSIM, depth_mask=None, 
                                                zero_shift=cfg.meta_info.zero_shift, cali=False, image_id=image_id)
                            
                            loss += lossValue.item()
                            psnr = psnr + sum(PSNR['PSNR_layer']) / len(PSNR['PSNR_layer']) if isinstance(PSNR['PSNR_layer'], list) else PSNR['PSNR_layer']
                            ssim = ssim + sum(SSIM['SSIM_layer']) / len(SSIM['SSIM_layer']) if isinstance(SSIM['SSIM_layer'], list) else SSIM['SSIM_layer']

                    loss = loss / (1 + batch_idx)
                    psnr = psnr / (1 + batch_idx)
                    ssim = ssim / (1 + batch_idx)
                    
                    # print the information of the validation
                    print("Eval loss: {}; psnr: {:2f}; ssim: {:4f}".format(loss, psnr, ssim))
                    open(log_name, 'a').write("Eval loss: {}; psnr: {:2f}; ssim: {:4f}\n".format(loss, psnr, ssim))

                    # save the best model and the latest model
                    if psnr >= cfg.meta_info.best_psnr:
                        cfg.meta_info.best_psnr = psnr
                        torch.save({
                            'global_step': epoch,   # save the global step
                            'network_fn_state_dict': net.state_dict(),  # save the network state dict
                            'optimizer_state_dict': optimizer.state_dict(),  # save the optimizer state dict
                        }, os.path.join(check_point_dir, 'Best_Loss.pth'))
                        print('Saved checkpoints at', os.path.join(check_point_dir, 'Best_Loss.pth'))

                    torch.save({
                            'global_step': epoch,   # save the global step
                            'network_fn_state_dict': net.state_dict(),  # save the network state dict
                            'optimizer_state_dict': optimizer.state_dict(),  # save the optimizer state dict
                        }, os.path.join(check_point_dir, 'model_latest.pth'))
                    print('Saved checkpoints at', os.path.join(check_point_dir, 'model_latest.pth'))

                    # viasualize the results in the tensorboard
                    utils.to_tensorboard(epoch,
                                        recon_field.abs(),
                                        recon_field.angle(),
                                        target,
                                        lr_now,
                                        loss,
                                        psnr,
                                        ssim,
                                        writer,
                                        'valid')

            writer.close()

    # evaluation / inference
    else:
        print('evaluation...')
        net.eval()   # set the network to the evaluation mode

        loss = 0.
        psnr = 0.
        ssim = 0.

        # define the folder for saving the evaluation results
        if cfg.meta_info.zero_shift:
            res_dir_eval = os.path.join(res_dir, 'eval_epoch{}_zeroshift'.format(cfg.meta_info.start))
        else:
            res_dir_eval = os.path.join(res_dir, 'eval_epoch{}'.format(cfg.meta_info.start))

        # pop a batch of data from the evaluation set
        for batch_idx, (data, target, image_id, ini_phase) in enumerate(exp_loader):

            image_id = image_id[0]  # get the image id
            image = data[:, 0:1, ...].to(device)  # transform the image from CPU to the GPUs
            ini_phase = ini_phase.to(device)

            # if the data is 3D data, generate the depth masks
            if cfg.data.test_dimension == "3D":
                depth = data[:, 1:2, ...].to(device)
                depth_mask = utils.generate_depth_stack_batch(cfg.data.dist_num, depth, 
                                                               cfg.data.slm_res, cfg.data.roi_res, 
                                                              torch.range(0, cfg.data.dist_num-1).to(device), 'tensor')  # shape: [batch, depth_num, slm_H, slm_W]

                # if the data is a RGBD 3D data, multiply the depth mask with the image
                # this will generated the sliced RGBD 3D data according to the depth mask
                if cfg.data.data_type == 'layer':
                    image = image * depth_mask  # shape: [batch, depth_num, slm_H, slm_W]
            else:
                depth = None
                depth_mask = None

            target = target.to(device)

            # during evaluation, the gradient graph is not needed
            with torch.no_grad():
                
                # predict the reconstruction field, the target reconstruction field, the SLM phase, and the input field
                recon_field, target_recon_field, slm_phase, input_field = render.predict(net, image, ini_phase, depth_map=depth)

                # calculate the loss, PSNR, and SSIM from the predicted reconstruction field and the target reconstruction field
                lossValue, PSNR, SSIM = render.evaluate(target, recon_field, target_recon_field, depth_mask=depth_mask, train=False, cali=False)

                loss += lossValue.item()
                psnr = psnr + sum(PSNR['PSNR_layer']) / len(PSNR['PSNR_layer']) if isinstance(PSNR['PSNR_layer'], list) else PSNR['PSNR_layer']
                ssim = ssim + sum(SSIM['SSIM_layer']) / len(SSIM['SSIM_layer']) if isinstance(SSIM['SSIM_layer'], list) else SSIM['SSIM_layer']

                # save the evaluation results
                render.save_results(batch_idx, recon_field, target_recon_field, slm_phase, input_field,
                                    cfg.meta_info.start, res_dir_eval, PSNR, SSIM, depth_mask=depth_mask, 
                                    zero_shift=cfg.meta_info.zero_shift, cali=False, image_id=image_id)

        
        loss = loss / (1 + batch_idx)
        psnr = psnr / (1 + batch_idx)
        ssim = ssim / (1 + batch_idx)
        print("Eval loss: {}; psnr: {:2f}; ssim: {:4f}".format(loss, psnr, ssim))


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