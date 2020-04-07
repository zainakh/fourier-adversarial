"""
Created on September 22
This code will run the baseline model on an adversarially corrupt (via FGSM) Fourier coefficient input. Comparisons will be made to one corrupted via Gaussian noise.
@author: Zain Khan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import readData as rd
import misc as sf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cwd = os.getcwd()
config = tf.compat.v1.ConfigProto()

# Model directory name
directory = 'baselineModel'
model_dir = cwd + '/' + directory
load_chk_pt = tf.train.latest_checkpoint(model_dir)

# Get data set
tst_org, tst_inp = rd.get_validation_data(num_from=20, num_img=1, acc=4)
mu = np.mean(tst_inp, axis=(-1, -2), keepdims=True)
st = np.std(tst_inp, axis=(-1, -2), keepdims=True)
tst_inp = (tst_inp - mu) / st
tst_inp = np.clip(tst_inp, -6, 6)
tst_org_adv = np.copy(tst_org)

# Corrupt the input Fourier coefficients with Gaussian noise
mean, std = mu[0][0][0], 0.2
gauss_noise = np.random.normal(mean, std, tst_inp.shape)
img_shape = np.shape(tst_inp)
avg_perturb = np.sum(np.absolute(gauss_noise)) / (img_shape[0] * img_shape[1] * img_shape[2])
tst_inp_gauss = np.copy(tst_inp) + gauss_noise

# Load model and predict
tst_org_adv = tst_org_adv[..., np.newaxis]
tst_inp = tst_inp[..., np.newaxis]
tst_rec = np.empty(tst_inp.shape, dtype=np.float32)
tst_inp_gauss = tst_inp_gauss[..., np.newaxis]
tst_rec_gauss = np.empty(tst_inp_gauss.shape, dtype=np.float32)
tst_inp_adv = np.copy(tst_inp)
tst_intr_adv = np.empty(tst_inp_adv.shape, dtype=np.float32)
tst_rec_adv = np.empty(tst_inp_adv.shape, dtype=np.float32)

# Initialize graph
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session(config=config) as sess:
    new_saver = tf.compat.v1.train.import_meta_graph(model_dir + '/modelTst.meta')
    new_saver.restore(sess, load_chk_pt)
    graph = tf.compat.v1.get_default_graph()
    predT = graph.get_tensor_by_name('predTst:0')
    senseT = graph.get_tensor_by_name('sense:0')

    # Run reconstruction for original and Gaussian noise samples
    tst_rec[0] = sess.run(predT, feed_dict={senseT: tst_inp[[0]]})
    tst_rec_gauss[0] = sess.run(predT, feed_dict={senseT: tst_inp_gauss[[0]]})

    # Get gradient of loss with respect to input
    origT = tf.convert_to_tensor(tst_org_adv)
    loss = tf.norm(predT - origT, ord='euclidean')
    g = tf.gradients(loss, senseT)
    sess.run(tf.global_variables_initializer())
    grad = sess.run(g, feed_dict={senseT: tst_inp[[0]]})

    # Create adversarial example and run reconstruction
    epsilon = avg_perturb
    sign_grad = np.sign(grad)
    tst_intr_adv = epsilon * sign_grad[0]
    tst_inp_adv[[0]] = tst_inp + tst_intr_adv
    tst_rec_adv[0] = sess.run(predT, feed_dict={senseT: tst_inp_adv[[0]]})


# Transform images into original shape
tst_inp, tst_rec = tst_inp[..., 0], tst_rec[..., 0]
tst_inp_gauss, tst_rec_gauss = tst_inp_gauss[..., 0], tst_rec_gauss[..., 0]
tst_inp_adv, tst_intr_adv, tst_rec_adv = tst_inp_adv[..., 0], tst_intr_adv[..., 0], tst_rec_adv[..., 0]
tst_inp, tst_rec = tst_inp * st + mu, tst_rec * st + mu
tst_inp_gauss, tst_rec_gauss = tst_inp_gauss * st + mu, tst_rec_gauss * st + mu
tst_inp_adv, tst_rec_adv = tst_inp_adv * st + mu, tst_rec_adv * st + mu

# Calculate L2 Norm errors
original_error = np.sqrt(np.sum((tst_rec[:, :, :] - tst_org[:, :, :]) ** 2))
gaussian_error = np.sqrt(np.sum((tst_rec_gauss[:, :, :] - tst_org[:, :, :]) ** 2))
adv_error = np.sqrt(np.sum((tst_rec_adv[:, :, :] - tst_org[:, :, :]) ** 2))

# Calculate SSIM and PSNR
psnr_orig = sf.psnr2(tst_org, tst_rec)
psnr_gauss = sf.psnr2(tst_org, tst_rec_gauss)
psnr_adv = sf.psnr2(tst_org, tst_rec_adv)

ssim_orig = sf.ssim2(tst_org, tst_rec)
ssim_gauss = sf.ssim2(tst_org, tst_rec_gauss)
ssim_adv = sf.ssim2(tst_org, tst_rec_adv)

# Display reconstruction error results
print('\nOriginal')
print('SSE : {}'.format(original_error))
print('PSNR : {}'.format(psnr_orig))
print('SSIM : {}\n'.format(ssim_orig))

print('Gaussian')
print('SSE : {}'.format(gaussian_error))
print('PSNR : {}'.format(psnr_gauss))
print('SSIM : {}\n'.format(ssim_gauss))

print('Adversarial')
print('SSE : {}'.format(adv_error))
print('PSNR : {}'.format(psnr_adv))
print('SSIM : {}'.format(ssim_adv))

# Subplot parameters
rows, cols = 3, 3

# Display original results
original_images = [tst_org, tst_inp, tst_rec]
original_titles = ['Original', 'FC', 'Reconstruct']
for i in range(1, 4):
    plt.subplot(rows, cols, i)
    plt.imshow(original_images[i - 1][0], cmap='gray')
    plt.title(original_titles[i - 1])
    plt.axis('off')

# Display Gaussian results
gaussian_images = [gauss_noise, tst_inp_gauss, tst_rec_gauss]
gaussian_titles = ['Mask', 'Gaussian FC', 'Gaussian Reconstruct']
for i in range(1, 4):
    plt.subplot(rows, cols, i + 3)
    plt.imshow(gaussian_images[i - 1][0], cmap='gray')
    plt.title(gaussian_titles[i - 1])
    plt.axis('off')

# Display adversarial results
adv_images = [tst_intr_adv, tst_inp_adv, tst_rec_adv]
adv_titles = ['Mask', 'Adv. FC', 'Adv. Reconstruct']
for i in range(1, 4):
    plt.subplot(rows, cols, i + 6)
    plt.imshow(adv_images[i - 1][0], cmap='gray')
    plt.title(adv_titles[i - 1])
    plt.axis('off')

plt.savefig('corrupt.png', bbox_inches='tight')

'''
from numpy import save, load
diff_sse = adv_error - original_error 
perturb = avg_perturb * (img_shape[0] * img_shape[1] * img_shape[2])
sse_vs_perturb = load('data_orig.npy')
sse_vs_perturb = np.append(sse_vs_perturb, [diff_sse, perturb, std])
save('data_orig.npy', sse_vs_perturb)
print(sse_vs_perturb)
'''

print(np.sum(np.absolute(tst_rec)))
