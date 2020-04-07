import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

sse_data_gauss = np.load('data.npy')
sse_data_orig = np.load('data_orig.npy')

# Group into threes 
# (Adv SSE - Other SSE, Perturb, STD in Gauss noise)
gauss_data = []
it = iter(sse_data_gauss)
for elem in it:
    gauss_data.append([elem, next(it), next(it)])
gauss_data[0], gauss_data[1] = gauss_data[1], gauss_data[0]
gauss_x = [item[1] for item in gauss_data]
gauss_y = [item[0] for item in gauss_data]

orig_data = []
it = iter(sse_data_orig)
for elem in it:
    orig_data.append([elem, next(it), next(it)])
orig_data[0], orig_data[1] = orig_data[1], orig_data[0]
orig_x = [item[1] for item in orig_data]
orig_y = [item[0] for item in orig_data]

plt.plot(gauss_x, gauss_y, marker='o', label='Gaussian Noise Reconstruction')
plt.plot(orig_x, orig_y, marker='o', color='r', label='Original Reconstruction')
plt.plot(orig_x, len(orig_y) * [0], color='black')
plt.ylabel('SSE Difference b/t Adversarial Reconstruction and Other Reconstruction')
plt.xlabel('Total Perturbations Added')
plt.legend(loc='lower left')
plt.savefig('sse_graph.png')