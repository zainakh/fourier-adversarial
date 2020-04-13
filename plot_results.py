import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

all_data = np.load('Apr13.npy')
n = 4
data = [all_data[k:k+n] for k in range(0, len(all_data), n)]
data = np.asarray(data)

# Group into fours 
# (Perturb, Original Error, Gaussian Error, Adversarial Error)
x = [item[0] for item in data]
orig_y = [item[1] for item in data]
gauss_y = [item[2] for item in data]
adv_y = [item[3] for item in data]

plt.plot(x, gauss_y, marker='o', label='Gaussian Noise Reconstruction')
plt.plot(x, adv_y, marker='o', color='b', label='Adversarial Reconstruction')
plt.plot(x, orig_y, marker='o', color='r', label='Original Reconstruction')
plt.ylabel('SSE in Reconstruction')
plt.xlabel('Total Perturbations Added')
plt.legend(loc='upper left')
plt.savefig('sse_graph.png')
