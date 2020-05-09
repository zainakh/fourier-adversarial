import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

all_data_partial = np.load('Apr27_partial.npy')
all_data_full = np.load('Apr27_full.npy')

n = 4
data_partial = [all_data_partial[k:k+n] for k in range(0, len(all_data_partial), n)]
data_full = [all_data_full[k:k+n] for k in range(0, len(all_data_full), n)]
data_partial = np.asarray(data_partial)
data_full = np.asarray(data_full)

# Group into fours 
# (Perturb, Original Error, Gaussian Error, Adversarial Error)
def split(data):
    x = [item[0] for item in data]
    orig_y = [item[1] for item in data]
    gauss_y = [item[2] for item in data]
    adv_y = [item[3] for item in data]

    return [x, orig_y, gauss_y, adv_y]

[x_p, p_orig, p_gauss, p_adv] = split(data_partial)
[x_f, f_orig, f_gauss, f_adv] = split(data_full)

plt.plot(x_p, p_gauss, marker='o', ls='--', color='seagreen', label='Gaussian Noise Partial Reconstruction')
plt.plot(x_p, p_adv, marker='o', ls='--', color='darkblue', label='Adversarial Partial Reconstruction')
plt.plot(x_p, p_orig, marker='o', ls='--', color='crimson', label='Original Partial Reconstruction')
plt.plot(x_f, f_gauss, marker='o', color='seagreen', label='Gaussian Noise Full Reconstruction')
plt.plot(x_f, f_adv, marker='o', color='darkblue', label='Adversarial Full Reconstruction')
plt.plot(x_f, f_orig, marker='o', color='crimson', label='Original Full Reconstruction')
plt.ylabel('SSE in Reconstruction')
plt.xlabel('Total Perturbations Added')
plt.title('Fourier Coefficient Reconstruction Error vs Perturbations Added')
plt.legend(loc='upper left')
plt.savefig('sse_graph_both.png')
