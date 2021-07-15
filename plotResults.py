import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

all_data_partial = np.load('perturbation-data/Jun12_p_psnr.npy')
all_data_full = np.load('perturbation-data/Jun12_f_psnr.npy')

n = 4
data_partial = [all_data_partial[k:k+n] for k in range(0, len(all_data_partial), n)]
data_full = [all_data_full[k:k+n] for k in range(0, len(all_data_full), n)]
data_partial = np.asarray(data_partial)
data_full = np.asarray(data_full)


# Group into fours 
# (Perturb, FC Error, Gaussian Error, Adversarial Error)
def split(data):
    x = [item[0] for item in data]
    orig_y = [item[1] for item in data]
    gauss_y = [item[2] for item in data]
    adv_y = [item[3] for item in data]

    return [x, orig_y, gauss_y, adv_y]


[x_p, p_orig, p_gauss, p_adv] = split(data_partial)
[x_f, f_orig, f_gauss, f_adv] = split(data_full)

fig, axs = plt.subplots(2, sharex=True)

axs[0].plot(x_f, f_orig, marker='o', color='crimson', label=r'$\hat{x}_1$')
axs[0].plot(x_f, f_gauss, marker='o', color='seagreen', label=r'$\hat{x}_3$')
axs[0].plot(x_f, f_adv, marker='o', color='darkblue', label=r'$\hat{x}_4$')
axs[0].set_xlabel('')
axs[0].legend(loc='upper right')

axs[1].plot(x_p, p_orig, marker='o', ls='--', color='crimson', label=r'$\hat{x}_2$')
axs[1].plot(x_p, p_gauss, marker='o', ls='--', color='seagreen', label=r'$\hat{x}_5$')
axs[1].plot(x_p, p_adv, marker='o', ls='--', color='darkblue', label=r'$\hat{x}_6$')


fig.text(0.07, 0.5, 'PSNR Score', va='center', ha='center', rotation='vertical', fontsize=12)
fig.text(0.5, 0.03, 'Amount of Perturbations Added as Fraction of the Full Signal', ha='center', fontsize=12)
fig.suptitle('Reconstruction PSNR vs Perturbations Added', x=0.5, y=0.92)
plt.legend(loc='upper right')
plt.savefig('psnr_plot.pdf')
