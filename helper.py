
import numpy as np
import matplotlib.pyplot as plt
import misc as sf
from os import path

def save_results(datafile, img_shape, original_error, gaussian_error, adv_error, fraction):
    from numpy import save, load

    errors = [fraction, original_error, gaussian_error, adv_error]
    if path.exists(datafile):
        sse_vs_perturb = load(datafile)
        sse_vs_perturb = np.append(sse_vs_perturb, errors)
        save(datafile, sse_vs_perturb)
    else:
        save(datafile, errors)
    print(errors)


def image_grid_small(filename, orig, perturb, final):
    rows, cols = 2, 3
    original_images = [orig, perturb, final]
    original_titles = ['FC', 'Perurbation Added', 'Combined FC']
    for i in range(1, 4):
        plt.subplot(rows, cols, i)
        plt.imshow(original_images[i - 1][0], cmap='gray')
        plt.title(original_titles[i - 1])
        plt.axis('off')

    # Display log of original results
    log_images = [np.log(orig), np.log(perturb), np.log(final)]
    log_titles = ['Log FC', 'Log Perturbation Added', 'Log Combined FC']
    for i in range(1, 4):
        plt.subplot(rows, cols, i + 3)
        plt.imshow(log_images[i - 1][0], cmap='gray')
        plt.title(log_titles[i - 1])
        plt.axis('off')
    
    plt.savefig(filename, bbox_inches='tight')


def save_images(filename_prefix, tst_org, tst_inp, fc_rec, gauss_noise, tst_inp_gauss, tst_rec_gauss, tst_intr_adv, tst_inp_adv, tst_rec_adv):
    # Display FC results
    original_images = [tst_org, tst_inp, fc_rec]
    original_titles = ['Original', 'FC_Perturbed', 'FC_Reconstruct']
    for i in range(1, 4):
        plt.imshow(original_images[i - 1][0], cmap='gray')
        plt.axis('off')
        plt.savefig(original_titles[i - 1])

    # Display Gaussian results
    gaussian_images = [gauss_noise, tst_inp_gauss, tst_rec_gauss]
    gaussian_titles = ['Gauss_Perturbation', 'Gaussian_Input_Image', 'Gaussian_Reconstruct']
    for i in range(1, 4):
        plt.imshow(gaussian_images[i - 1][0], cmap='gray')
        plt.axis('off')
        plt.savefig(gaussian_titles[i - 1])

    # Display adversarial results
    adv_images = [tst_intr_adv, tst_inp_adv, tst_rec_adv]
    adv_titles = ['Adv_Perturbation', 'Adv_Input_Image', 'Adv_Reconstruct']
    for i in range(1, 4):
        plt.imshow(adv_images[i - 1][0], cmap='gray')
        plt.axis('off')
        plt.savefig(adv_titles[i - 1])


def image_grid(filename, tst_org, tst_inp, fc_rec, gauss_noise, tst_inp_gauss, tst_rec_gauss, tst_intr_adv, tst_inp_adv, tst_rec_adv):
    # Subplot parameters
    rows, cols = 3, 3

    # Display original results
    original_images = [tst_org, tst_inp, fc_rec]
    original_titles = ['Original', 'FC', 'FC Reconstruct']
    for i in range(1, 4):
        plt.subplot(rows, cols, i)
        plt.imshow(original_images[i - 1][0], cmap='gray')
        plt.title(original_titles[i - 1])
        plt.axis('off')

    # Display Gaussian results
    gaussian_images = [gauss_noise, tst_inp_gauss, tst_rec_gauss]
    gaussian_titles = ['Perturbation', 'Gaussian FC', 'Gaussian Reconstruct']
    for i in range(1, 4):
        plt.subplot(rows, cols, i + 3)
        plt.imshow(gaussian_images[i - 1][0], cmap='gray')
        plt.title(gaussian_titles[i - 1])
        plt.axis('off')

    # Display adversarial results
    adv_images = [tst_intr_adv, tst_inp_adv, tst_rec_adv]
    adv_titles = ['Perturbation', 'Adv. FC', 'Adv. Reconstruct']
    for i in range(1, 4):
        plt.subplot(rows, cols, i + 6)
        plt.imshow(adv_images[i - 1][0], cmap='gray')
        plt.title(adv_titles[i - 1])
        plt.axis('off')

    plt.savefig(filename, bbox_inches='tight')


def print_all_errors(tst_org, tst_rec, tst_rec_gauss, tst_rec_adv):
    original_error, gaussian_error, adv_error = calculate_sse(tst_org, tst_rec, tst_rec_gauss, tst_rec_adv)
    psnr_ssim = calculate_psnr_ssim(tst_org, tst_rec, tst_rec_gauss, tst_rec_adv)
    psnr_orig, psnr_gauss, psnr_adv = psnr_ssim[0]
    ssim_orig, ssim_gauss, ssim_adv = psnr_ssim[1]

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


def calculate_psnr_ssim(tst_org, tst_rec, tst_rec_gauss, tst_rec_adv):
    psnr_orig = sf.psnr2(tst_org, tst_rec)
    psnr_gauss = sf.psnr2(tst_org, tst_rec_gauss)
    psnr_adv = sf.psnr2(tst_org, tst_rec_adv)

    ssim_orig = sf.ssim2(tst_org, tst_rec)
    ssim_gauss = sf.ssim2(tst_org, tst_rec_gauss)
    ssim_adv = sf.ssim2(tst_org, tst_rec_adv)

    return ([psnr_orig, psnr_gauss, psnr_adv], [ssim_orig, ssim_gauss, ssim_adv])


def calculate_sse(tst_org, tst_fc, tst_rec_gauss, tst_rec_adv):
    fc_error = np.sqrt(np.sum((tst_fc[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
    gaussian_error = np.sqrt(np.sum((tst_rec_gauss[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
    adv_error = np.sqrt(np.sum((tst_rec_adv[:, :, :] - np.copy(tst_org)[:, :, :]) ** 2))
    return [fc_error, gaussian_error, adv_error]

def calculate_psnr(tst_org, tst_fc, tst_rec_gauss, tst_rec_adv):
    psnr_fc = sf.psnr2(tst_org, tst_fc)
    psnr_gauss = sf.psnr2(tst_org, tst_rec_gauss)
    psnr_adv = sf.psnr2(tst_org, tst_rec_adv)
    return [psnr_fc, psnr_gauss, psnr_adv]


def calculate_ssim(tst_org, tst_fc, tst_rec_gauss, tst_rec_adv):
    ssim_fc = sf.ssim2(tst_org, tst_fc)
    ssim_gauss = sf.ssim2(tst_org, tst_rec_gauss)
    ssim_adv = sf.ssim2(tst_org, tst_rec_adv)
    return [ssim_fc, ssim_gauss, ssim_adv]