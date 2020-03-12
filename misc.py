"""
Created on 6 August, 2018
This file contains some supporting functions used during training and testing.
@author: Hemant Aggarwal
"""

import os
import numpy as np
import mkl_fft
from skimage.measure import compare_ssim, compare_psnr

os.environ['OMP_NUM_THREADS'] = '8'

def crop(data, shape=(320, 320)):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def fft2c(img):
    shp = img.shape
    nimg = int(np.prod(shp[0:-2]))
    scale = 1/np.sqrt(np.prod(shp[-2:]))
    img = np.reshape(img, (nimg, shp[-2], shp[-1]))

    tmp = np.empty_like(img,dtype=np.complex64)
    for i in range(nimg):
        tmp[i] = scale*np.fft.fftshift(mkl_fft.fft2(np.fft.ifftshift(img[i])))

    kspace = np.reshape(tmp, shp)
    return kspace


def ifft2c(kspace):
    shp = kspace.shape
    scale = np.sqrt(np.prod(shp[-2:]))
    nimg = int(np.prod(shp[0:-2]))

    kspace = np.reshape(kspace, (nimg, shp[-2], shp[-1]))

    tmp = np.empty_like(kspace)
    for i in range(nimg):
        tmp[i] = scale*np.fft.fftshift(mkl_fft.ifft2(np.fft.ifftshift(kspace[i])))

    return np.reshape(tmp, shp)


def sos(data, dim=-3):
    return np.sqrt(np.sum(np.abs(data)**2, dim))


def ssim2(gt, pred):
    return compare_ssim(gt.transpose(1, 2, 0), pred.transpose(1, 2, 0),
                        multichannel=True, data_range=gt.max())


def psnr2(gt, pred):
    return compare_psnr(gt, pred, data_range=gt.max())


def r2c(inp):
    return inp[..., 0] + 1j*inp[..., 1]


def c2r(inp):
    return np.stack([np.real(inp), np.imag(inp)], axis=-1)
