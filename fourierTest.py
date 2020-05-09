
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import readData as rd
import misc as sf
from helper import *


perturb = 2000

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

cwd = os.getcwd()
config = tf.compat.v1.ConfigProto()

# Model directory name
directory = 'baselineModel'
model_dir = cwd + '/' + directory
load_chk_pt = tf.train.latest_checkpoint(model_dir)

tst_org, tst_inp = rd.get_validation_data(num_from=20, num_img=1, acc=4, full=False) 
mu = np.mean(tst_inp, axis=(-1, -2), keepdims=True)
st = np.std(tst_inp, axis=(-1, -2), keepdims=True)
tst_inp = (tst_inp - mu) / st

# Initialize graph
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session(config=config) as sess:
    new_saver = tf.compat.v1.train.import_meta_graph(model_dir + '/modelTst.meta')
    new_saver.restore(sess, load_chk_pt)
    graph = tf.compat.v1.get_default_graph()
    predT = graph.get_tensor_by_name('predTst:0')
    senseT = graph.get_tensor_by_name('sense:0')

    tst_coeff = sf.fft2c(tst_org)
    full_fc = tf.convert_to_tensor(tst_coeff, dtype=tf.complex64)
    inp_fc = tf.convert_to_tensor(tst_inp, dtype=tf.complex64)
    loss_fc = tf.norm(full_fc - inp_fc, ord='euclidean')
    grad_ys = tf.ones(loss_fc.shape, dtype=tf.complex64)
    g = tf.gradients(loss_fc, inp_fc, grad_ys=grad_ys)
    g.eval(session=sess)
    #grad = sess.run(g, feed_dict={inp_fc:tst_inp, full_fc:tst_coeff})

    inp_fc += np.copy(grad) / 320

    adv = sf.ifft2c(inp_fc)

    rows, cols = 1, 2

    # Display original results
    fc_images = [tst_org, adv]
    fc_titles = ['FC Full', 'Adv']
    for i in range(1, 2):
        plt.subplot(rows, cols, i)
        plt.imshow(fc_images[i - 1][0], cmap='gray')
        plt.title(fc_titles[i - 1])
        plt.axis('off')

    plt.savefig('fc_attack', bbox_inches='tight')

    fc_error = np.sqrt(np.sum((adv[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
    print(fc_error)
