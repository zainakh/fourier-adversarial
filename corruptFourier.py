"""
Created on September 22
This code will run the baseline model on an adversarially corrupt (via FGSM) Fourier coefficient input. Comparisons will be made to one corrupted via Gaussian noise.
@author: Zain Khan
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import readData as rd
import misc as sf
import sys
from helper import save_results, image_grid, image_grid_small, calculate_sse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="filename for saved numpy data")
    parser.add_argument("-i", "--image", help="filename for saved image")
    parser.add_argument("-fc", "--fullcoeff", action="store_true", help="if full set of Fourier coefficients is desired")
    parser.add_argument("-p", "--partial", dest="fullcoeff", action="store_false", help="if partial set of Fourier coefficients is desired")
    return parser.parse_args()


def main(argv):
    filename = argv.filename
    img_name = argv.image
    full = argv.fullcoeff

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '8'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    cwd = os.getcwd()
    config = tf.compat.v1.ConfigProto()

    # Model directory name
    directory = 'baselineModel'
    model_dir = cwd + '/' + directory
    load_chk_pt = tf.train.latest_checkpoint(model_dir)

    #epsilons = np.asarray([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])
    epsilons = np.asarray([0, 2.5, 5])
    perturb_sizes = list(np.linspace(0, 30000, num=9))
    epsilons_perturb = zip(epsilons, perturb_sizes)
    print(perturb_sizes)

    for epsilon, perturb in epsilons_perturb:
        tst_org, tst_inp = rd.get_validation_data(num_from=20, num_img=1, acc=4, full=full) 

        mu = np.mean(tst_org, axis=(-1, -2), keepdims=True)
        st = np.std(tst_org, axis=(-1, -2), keepdims=True)
        tst_inp = (tst_inp - mu) / st
        tst_inp = np.clip(tst_inp, -6, 6)
        tst_org_adv = np.copy(tst_org)
        tst_inp_orig = np.copy(tst_inp)

        # Load model and predict
        tst_org_adv = tst_org_adv[..., np.newaxis]
        tst_inp = tst_inp[..., np.newaxis]
        tst_rec = np.zeros(tst_inp.shape, dtype=np.float32)
        tst_inp_adv = np.copy(tst_inp)
        tst_intr_adv = np.zeros(tst_inp_adv.shape, dtype=np.float32)
        tst_rec_adv = np.zeros(tst_inp_adv.shape, dtype=np.float32)

        # Initialize graph
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session(config=config) as sess:
            new_saver = tf.compat.v1.train.import_meta_graph(model_dir + '/modelTst.meta')
            new_saver.restore(sess, load_chk_pt)
            graph = tf.compat.v1.get_default_graph()
            predT = graph.get_tensor_by_name('predTst:0')
            senseT = graph.get_tensor_by_name('sense:0')

            # Run reconstruction for original reconstruction
            tst_rec[0] = sess.run(predT, feed_dict={senseT: tst_inp[[0]]})

            # Get gradient of loss with respect to input
            origT = tf.convert_to_tensor(tst_org_adv)
            loss = tf.norm(origT - predT, ord='euclidean')
            g = tf.gradients(loss, senseT)
            grad = sess.run(g, feed_dict={senseT: tst_inp[[0]]})

            # Create adversarial example and run reconstruction
            tst_intr_adv = epsilon * np.copy(grad)[0]
            if epsilon != 0:
                tst_intr_adv *=  perturb / (np.sum(np.abs(tst_intr_adv)))            
                fraction = np.sum(np.abs(tst_inp_adv)) / np.sum(np.abs(tst_intr_adv))
            else:
                fraction = 0

            # Fraction of original image is perturbation
            print('Fraction', fraction)
            tst_inp_adv += np.copy(tst_intr_adv)
            tst_rec_adv[0] = sess.run(predT, feed_dict={senseT: tst_inp_adv[[0]]})

            # Corrupt the input Fourier coefficients with Gaussian noise
            mean, std = mu[0][0][0], st
            gauss_noise = np.random.normal(mean, std, tst_inp_orig.shape)
            gauss_noise = gauss_noise * (np.sum(np.absolute(tst_intr_adv)) / np.sum(np.absolute(gauss_noise)))
            img_shape = np.shape(tst_inp_orig)
            #avg_perturb = np.sum(np.absolute(gauss_noise)) / (img_shape[0] * img_shape[1] * img_shape[2])
            tst_inp_gauss = np.copy(tst_inp_orig) - np.copy(gauss_noise)
            tst_inp_gauss = tst_inp_gauss[..., np.newaxis]
            tst_rec_gauss = np.zeros(tst_inp_gauss.shape, dtype=np.float32)
            
            # Run reconstruction for Gaussian case
            tst_rec_gauss[0] = sess.run(predT, feed_dict={senseT: tst_inp_gauss[[0]]})

        # Transform images into original shape
        tst_inp = tst_inp[..., 0]
        tst_rec = tst_rec[..., 0]
        tst_inp_gauss = tst_inp_gauss[..., 0]
        tst_rec_gauss = tst_rec_gauss[..., 0]
        tst_inp_adv = tst_inp_adv[..., 0]
        tst_intr_adv = tst_intr_adv[..., 0]
        tst_rec_adv = tst_rec_adv[..., 0]
        tst_inp, tst_rec = tst_inp * st + mu, tst_rec * st + mu
        tst_inp_gauss, tst_rec_gauss = tst_inp_gauss * st + mu, tst_rec_gauss * st + mu
        tst_inp_adv, tst_rec_adv = tst_inp_adv * st + mu, tst_rec_adv * st + mu


        # Add noise to FC to see how it reacts
        coeff_inp = sf.fft2c(tst_inp)
        first_img = sf.sos(sf.crop(sf.ifft2c(coeff_inp), (320, 320)))
        first_img = first_img[np.newaxis, ...]

        shp = list(coeff_inp.shape)

        # Scale should be 320, so dividing by it in the Fourier space will yield correct size in image space
        scale = np.sqrt(np.prod(shp[-2:]))
        gauss_noise_fc_real = np.random.normal(mean, std, size=shp)
        gauss_noise_fc_imag = np.random.normal(mean, std, size=shp) * 1j
        gauss_noise_fc = gauss_noise_fc_real[...] + gauss_noise_fc_imag[...]
        if epsilon != 0:  
            #fc_scaling = (perturb) / (np.sum(np.abs(gauss_noise_fc)) * scale)
            fc_scaling = np.sum(np.abs(coeff_inp)) / fraction
            gauss_noise_fc *= fc_scaling
            print(perturb, scale, perturb/scale, np.sum(np.abs(gauss_noise_fc)), np.sum(np.abs(coeff_inp)))
            print('FC Scaling', fc_scaling)
        else:
            gauss_noise_fc = np.zeros(shape=gauss_noise_fc.shape, dtype=np.complex64)

        #coeff_final = np.copy(coeff_inp) + np.copy(gauss_noise_fc)
        coeff_final = np.copy(coeff_inp)
        idx_max = np.unravel_index(coeff_final.argmax(), coeff_final.shape)
        coeff_final[idx_max[0]][idx_max[1]][idx_max[2]] += np.sum(np.abs(gauss_noise_fc))

        print('Amax', np.amax(coeff_final))
        tst_fc = sf.sos(sf.crop(sf.ifft2c(coeff_final), (320, 320)))

        tst_fc_noise = sf.sos(sf.crop(sf.ifft2c(gauss_noise_fc), (320, 320)))
        tst_fc_noise = tst_fc_noise[np.newaxis, ...]

        print(np.sum(np.abs(tst_inp)), np.sum(np.abs(first_img)), np.sum(np.abs(tst_fc)))

        tst_fc = tst_fc[np.newaxis, ...]
        print('PIC SUMS', epsilon, np.sum(np.abs(tst_fc[...])), np.sum(np.abs(tst_rec_gauss)), np.sum(np.abs(tst_rec_adv[...])))
        print('NOISE SUMS', np.sum(np.abs(tst_fc_noise)), np.sum(np.abs(gauss_noise)), np.sum(np.abs(tst_intr_adv)))

        [fc_error, gaussian_error, adv_error] = calculate_sse(tst_org, tst_fc, tst_rec_gauss, tst_rec_adv)

        if fraction != 0:
            image_grid_small('fc_visual', np.abs(coeff_inp), np.abs(gauss_noise_fc), np.abs(coeff_final))
            fraction_original_to_perturb = 1 / fraction
        else:
            fraction_original_to_perturb = 0

        save_results(filename + '.npy', img_shape, fc_error, gaussian_error, adv_error, fraction_original_to_perturb)

        if img_name:
            image_grid(img_name, tst_org, tst_inp, tst_fc, gauss_noise, tst_inp_gauss, tst_rec_gauss, tst_intr_adv, tst_inp_adv, tst_rec_adv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))