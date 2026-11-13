import cv2
import numpy as np
import os
import shutil
import datetime

def make_dir(path, overwrite=False):
    '''
    This function creates a directory if it does not exist. If it exists, 
    it will delete the directory and create a new one if overwrite is set to True.
    Args:
        path (str): Path to the directory.
        overwrite (bool): If True, it will delete the directory and create a new one. Default is False.
    '''

    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.mkdir(path)
    elif os.path.exists(path) and overwrite:
        shutil.rmtree(path)
        os.mkdir(path)


def format_time(time):
    elapsed_rounded = int(round((time)))
    # hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def to_tensorboard(epoch, recon_amp, recon_phase, target, lr, loss, psnr, ssim, writer, train_valid):
    '''
    This function writes the images and metrics to tensorboard.
    Args:
        epoch (int): Current epoch.
        recon_amp (torch.Tensor): Reconstructed amplitude.
        recon_phase (torch.Tensor): Reconstructed phase.
        target (torch.Tensor): Target image.
        lr (float): Learning rate.
        loss (float): Loss value.
        psnr (float): PSNR value.
        ssim (float): SSIM value.
        writer (SummaryWriter): SummaryWriter object.
        train_valid (str): 'train' or 'valid'.
    '''

    recon_amp = recon_amp.cpu().detach().numpy()[0,0,...]
    recon_amp = cv2.normalize(recon_amp.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    recon_amp = recon_amp[np.newaxis,...]

    recon_phase = recon_phase.cpu().detach().numpy()[0,0,...]
    recon_phase = (recon_phase + np.pi) % (2 * np.pi) / (2 * np.pi)
    recon_phase = recon_phase[np.newaxis,...]

    target = target.cpu().detach().numpy()[0,0,...]
    target = cv2.normalize(target.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    target = target[np.newaxis,...]

    writer.add_image(train_valid + '/amplitude', recon_amp, epoch)

    writer.add_image(train_valid + '/phase', recon_phase, epoch)

    writer.add_image(train_valid + '/target', target, epoch)

    writer.add_scalar(train_valid + '/lr', lr, epoch)
    writer.add_scalar(train_valid + '/loss', loss, epoch)
    writer.add_scalar(train_valid + '/psnr', psnr, epoch)
    writer.add_scalar(train_valid + '/ssim', ssim, epoch)