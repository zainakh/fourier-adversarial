"""
Created on Wed Sep 16
This code will run the baseline model on a validation slice without any change.
@author: haggarwal
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '8'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import misc as sf
import readData as rd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
cwd = os.getcwd()
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

# %% name of model directory
directory = 'baselineModel'
modelDir = cwd + '/' + directory
loadChkPoint = tf.train.latest_checkpoint(modelDir)

# %% get the dataset
tstOrg, tstInp = rd.get_validation_data(num_from=20, num_img=1, acc=4)
mu = np.mean(tstInp, axis=(-1, -2), keepdims=True)
st = np.std(tstInp, axis=(-1, -2), keepdims=True)
tstInp = (tstInp - mu) / st
tstInp = np.clip(tstInp, -6, 6)

nImg, nRow, nCol = tstOrg.shape


# %% load the model and do the prediction
tstInp = tstInp[..., np.newaxis]
tstRec = np.empty(tstInp.shape, dtype=np.float32)
tstGrad = np.empty(tstInp.shape, dtype=np.float32)
tf.reset_default_graph()
with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(modelDir + '/modelTst.meta')
    new_saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    predT = graph.get_tensor_by_name('predTst:0')
    senseT = graph.get_tensor_by_name('sense:0')
    for i in range(nImg):
        dataDict = {senseT: tstInp[[i]]}
        g = tf.gradients(predT, senseT)
        tstRec[i] = sess.run(predT, feed_dict=dataDict)

        sess.run(tf.global_variables_initializer())
        grad = sess.run(g, feed_dict=dataDict)

        
print(tstRec[i])
print(grad)

# %% calculate the PSNR and SSIM
tstRec = tstRec[..., 0]
tstInp = tstInp[..., 0]
tstRec = tstRec * st + mu
tstInp = tstInp * st + mu

psnrAtb = sf.psnr2(tstOrg, tstInp)
ssimAtb = sf.ssim2(tstOrg, tstInp)

psnrRec = sf.psnr2(tstOrg, tstRec)
ssimRec = sf.ssim2(tstOrg, tstRec)
# %
print('*************************************************')
print('  ' + 'Noisy ' + 'Rec')
print('  {0:.2f} {1:.2f}'.format(psnrAtb, psnrRec))
print('  {0:.2f} {1:.2f}'.format(ssimAtb, ssimRec))
print('*************************************************')

# %% display the results
plt.subplot(131)
plt.imshow(tstOrg[0], cmap='gray')
plt.title('orginal')
plt.axis('off')
plt.subplot(132)
plt.imshow(tstInp[0], cmap='gray')
plt.title('Input')
plt.axis('off')
plt.subplot(133)
plt.imshow(tstRec[0], cmap='gray')
plt.title('Output')
plt.axis('off')
plt.savefig('demo.png', bbox_inches='tight')
# plt.show()
# %%
