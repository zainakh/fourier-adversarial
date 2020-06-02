
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
mean = 0
std = 1/320

size = 320
tst_org = sf.crop(tst_org, (size, size))
tst_inp = sf.crop(tst_inp, (size, size))

# Add noise to FC to see how it reacts
coeff_inp = sf.fft2c(tst_inp)

shp = list(coeff_inp.shape)

# Use a set fraction of FC to perturb (1/fraction is the amount that will be perturbed)
fraction = 20

# Scale should be 320, so dividing by it in the Fourier space will yield correct size in image space
scale = np.sqrt(np.prod(shp[-2:]))
gauss_noise_fc_real = np.abs(np.random.normal(mean, std, size=shp))
gauss_noise_fc_imag = np.abs(np.random.normal(mean, std, size=shp) * 1j)
gauss_noise_fc = gauss_noise_fc_real[...] + gauss_noise_fc_imag[...]

#fc_scaling = np.sum(np.abs(coeff_inp)) / fraction
#gauss_noise_fc *= fc_
#print(fraction, scale, np.sum(np.abs(gauss_noise_fc)), np.sum(np.abs(coeff_inp)))

coeff_final = gauss_noise_fc
tst_fc = sf.ifft2c(coeff_final)
print('Coeff Norm', np.linalg.norm(coeff_final))
print('Gauss Norm', np.linalg.norm(tst_fc))


# Add perturbations to max value of FC (which is the DC component)
#coeff_final = np.copy(coeff_inp) + np.copy(gauss_noise_fc)
coeff_final = np.copy(coeff_inp) - np.copy(coeff_inp)
coeff_final[0][20][220] += gauss_noise_fc[0][17][228]
coeff_final[0][20][221] += gauss_noise_fc[0][49][228]
coeff_final[0][20][222] += gauss_noise_fc[0][62][228]
print('gauss number', gauss_noise_fc[0][17][228])


#coeff_final = coeff_final - coeff_final

#coeff_final[0][0][0] = 0.5
tst_fc = sf.ifft2c(coeff_final)
print('Coeff Norm', np.linalg.norm(coeff_final))
print('DC Norm', np.linalg.norm(tst_fc))
norm1 = np.linalg.norm(tst_fc)**2
print(np.linalg.norm(tst_fc)**2)




'''
# Add perturbations to max value of FC (which is the DC component)
coeff_final = np.copy(coeff_inp)
coeff_final = coeff_final - coeff_final
coeff_final[0][0][0] = 0.5
coeff_final[0][1][0] = 0.5
tst_fc = sf.ifft2c(coeff_final)
tst_fc = tst_fc[np.newaxis, ...]
tst_fc3 = tst_fc
fc_error = np.sqrt(np.sum((np.copy(tst_fc)[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
print('2nd SSE', fc_error)
print('Sum 2nd', np.sum(np.abs(tst_fc)))
norm3 = np.linalg.norm(tst_fc)**2
print(np.linalg.norm(tst_fc)**2)

res = np.linalg.norm(tst_fc1 + tst_fc2 - tst_fc3)
print(res)
'''

