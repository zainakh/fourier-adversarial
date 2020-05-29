
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import readData as rd
import misc as sf
import mkl_fft
from helper import *


# Testing the FC method with this inp array shows that it behaves as a Fourier transform
# It applies scaling afterwards (to check, just applying fft2c and ifft2c will show they are transforms of each other)
# Also tested with the identity matrix and it behaved as expected
inp = np.eye(4)
inp[0][0] -= 1
inp[2][2] -= 1
inp[3][3] -= 1

#print(inp)
#res = sf.fft2c(inp)
#print(res)
#print(sf.ifft2c(res))


tst_org, tst_inp = rd.get_validation_data(num_from=20, num_img=1, acc=4, full=True) 
mu = np.mean(tst_org, axis=(-1, -2), keepdims=True)
mean = mu[0][0][0]
std = np.std(tst_org, axis=(-1, -2), keepdims=True)

# Add noise to FC to see how it reacts
coeff_inp = sf.fft2c(tst_inp)
first_img = sf.sos(sf.crop(sf.ifft2c(coeff_inp), (320, 320)))
first_img = first_img[np.newaxis, ...]

shp = list(coeff_inp.shape)

# Use a set fraction of FC to perturb (1/fraction is the amount that will be perturbed)
fraction = 20

# Scale should be 320, so dividing by it in the Fourier space will yield correct size in image space
scale = np.sqrt(np.prod(shp[-2:]))
gauss_noise_fc_real = np.abs(np.random.normal(mean, std, size=shp))
gauss_noise_fc_imag = np.abs(np.random.normal(mean, std, size=shp) * 1j)
gauss_noise_fc = gauss_noise_fc_real[...] + gauss_noise_fc_imag[...]

fc_scaling = np.sum(np.abs(coeff_inp)) / fraction
gauss_noise_fc *= fc_scaling
#print(fraction, scale, np.sum(np.abs(gauss_noise_fc)), np.sum(np.abs(coeff_inp)))

# Add perturbations to the FC via the Gaussian method
coeff_final = np.copy(coeff_inp) + np.copy(gauss_noise_fc)
tst_fc = sf.sos(sf.crop(sf.ifft2c(coeff_final), (320, 320)))
tst_fc = tst_fc[np.newaxis, ...]
fc_error = np.sqrt(np.sum((np.copy(tst_fc)[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
print('Gaussian SSE', fc_error)
print('Sum Gaussian', np.sum(np.abs(tst_fc)))

# Add perturbations to max value of FC (which is the DC component)
coeff_final = np.copy(coeff_inp)
idx_max = np.unravel_index(coeff_final.argmax(), coeff_final.shape)
coeff_final[idx_max[0]][idx_max[1]][idx_max[2]] += np.sum(np.abs(gauss_noise_fc))
tst_fc = sf.sos(sf.crop(sf.ifft2c(coeff_final), (320, 320)))
tst_fc = tst_fc[np.newaxis, ...]
fc_error = np.sqrt(np.sum((np.copy(tst_fc)[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
print('DC SSE', fc_error)
print('Sum DC', np.sum(np.abs(tst_fc)))
