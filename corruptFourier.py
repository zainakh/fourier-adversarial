"""
Created on September 22
This code will run the baseline model on a corrupt Fourier coefficient input.
@author: Zain Khan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import readData as rd
from cleverhans.compat import reduce_sum
from cleverhans import utils_tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'

tf.executing_eagerly()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cwd = os.getcwd()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

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

_, nRow, nCol = tst_org.shape

# Corrupt the input Fourier coefficients with Gaussian noise
mean, std = 0.0, 0.5
gauss_noise = np.random.normal(mean, std, tst_inp.shape)
tst_inp_gauss = np.copy(tst_inp) + gauss_noise

# Function c2r concatenate complex input as new axis two two real inputs
c2r = lambda x: tf.stack([tf.real(x), tf.imag(x)], axis=-1)

# r2c takes the last dimension of real input and converts to complex
r2c = lambda x: tf.complex(x[..., 0], x[..., 1])


class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, csm, mask, lam):
        with tf.name_scope('Ainit'):
            s = tf.shape(mask)
            self.nrow, self.ncol = s[0], s[1]
            self.pixels = self.nrow*self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.to_float(self.pixels)), 0.)
            self.lam = lam

    def myAtA(self, img):
        with tf.name_scope('AtA'):
            coilImages = self.csm*img
            kspace = tf.fft2d(coilImages)/self.SF
            temp = kspace*self.mask
            coilImgs = tf.ifft2d(temp)*self.SF
            coilComb = tf.reduce_sum(coilImgs*tf.conj(self.csm), axis=0)
            coilComb = coilComb + self.lam * img
        return coilComb


def myCG(A,rhs):
    """
    This is my implementation of CG algorithm in TensorFlow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    rhs = r2c(rhs)
    cond = lambda i, rTr, *_: tf.logical_and(tf.less(i, 10), rTr > 1e-10)

    def body(i, rTr, x, r, p):
        with tf.name_scope('cgBody'):
            Ap = A.myAtA(p)
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p)*Ap))
            alpha = tf.complex(alpha, 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float( tf.reduce_sum(tf.conj(r)*r))
            beta = rTrNew / rTr
            beta = tf.complex(beta,0.)
            p = r + beta * p
        return i+1, rTrNew, x, r, p

    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = tf.to_float( tf.reduce_sum(tf.conj(r)*r),)
    loopVar=i,rTr,x,r,p
    out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return c2r(out)


def get_lambda():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=.05)
    return lam


def callCG(rhs):
    """
    this function will call the function myCG on each image in a batch
    """
    G = tf.get_default_graph()
    getnext = G.get_operation_by_name('getNext')
    _, _, csm, mask = getnext.outputs
    l = get_lambda()
    l2 = tf.complex(l, 0.)

    def fn(tmp):
        c, m, r = tmp
        Aobj = Aclass(c, m, l2)
        y = myCG(Aobj, r)
        return y
    inp = (csm, mask, rhs)
    rec = tf.map_fn(fn, inp, dtype=tf.float32, name='mapFn2')
    return rec


@tf.custom_gradient
def dc_manual_gradient(x):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjugate gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y = callCG(x)

    def grad(inp):
        out = callCG(inp)
        return out
    return y, grad


# Load model and predict
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

    # Find Gradient
    tst_rec_adv[0] = sess.run(predT, feed_dict={senseT: tst_inp_adv[[0]]})
    loss = tf.norm(predT - senseT, 2)

    '''
    custom_grad = dc_manual_gradient(senseT)
    gradient = tf.gradients(custom_grad, senseT)
    a = sess.run(gradient, feed_dict={senseT: tst_inp_adv[[0]]})

    print(a)

    eps = 0.3
    red_ind = list(range(1, len(gradient.get_shape())))
    avoid_zero_div = 1e-12

    square = tf.maximum(avoid_zero_div,
                        reduce_sum(tf.square(gradient),
                                   reduction_indices=red_ind,
                                   keepdims=True))
    optimal_perturbation = gradient / tf.sqrt(square)
    scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)

    # Add perturbation to original example to obtain adversarial example
    adv_x = senseT + scaled_perturbation

    tst_intr_adv[0] = sess.run(adv_x, feed_dict={senseT: tst_inp_adv[[0]]})
    tst_rec_adv[0] = sess.run(predT, feed_dict={senseT: tst_intr_adv[[0]]})
    '''


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

# Display error results
print('Original SSE Values')
print('{}\n'.format(original_error))

print('Gaussian SSE Values')
print('{}\n'.format(gaussian_error))

print('Adversarial SSE Values')
print('{}'.format(adv_error))

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
adv_noise = tst_intr_adv[:, :, :] - tst_inp_adv[:, :, :]
adv_images = [tst_org, tst_inp_adv, tst_rec_adv]
adv_titles = ['Mask', 'Adv. FC', 'Adv. Reconstruct']
for i in range(1, 4):
    plt.subplot(rows, cols, i + 6)
    plt.imshow(adv_images[i - 1][0], cmap='gray')
    plt.title(adv_titles[i - 1])
    plt.axis('off')

plt.savefig('corrupt.png', bbox_inches='tight')
