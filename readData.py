"""
Created on Monday, October 29
@author: Hemant Aggarwal
"""

import numpy as np
import h5py as h5
import misc as sf
import tensorflow as tf
from subsample import MaskFunc

np.set_printoptions(suppress=True, precision=4)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


def get_validation_data(num_from, num_img, acc, filename='file1000277.h5', full=False):
    if acc == 4:
        fraction = .08
    elif acc == 8:
        fraction = .04

    with h5.File(filename, 'r') as f:
        ksp = f['kspace'][num_from:num_from + num_img, :]
        org = f['reconstruction_rss'][num_from:num_from + num_img, :]

    nslc, _, rows, cols = ksp.shape
    mm = MaskFunc([fraction], [acc])
    msk = mm((1, cols, 1), 0)

    if full:
        msk = np.array([True for val in msk])
    
    msk = np.repeat(msk[np.newaxis], rows, axis=0)
    msk = np.repeat(msk[np.newaxis], nslc, axis=0)
    msk = msk.astype(np.complex64)
    b = ksp * msk[:, np.newaxis]
    inp = sf.sos(sf.crop(sf.ifft2c(b), (320, 320)))

    return org, inp
