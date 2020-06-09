import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')

partial = np.load('noise_transform.npy')
#full = np.load('f.npy')
x = np.arange(len(partial))

plt.plot(x, partial, marker='o', ls='--', color='seagreen', label='Partial FC')
#plt.plot(x, full, marker='o', color='darkblue', label='Full FC')
plt.ylabel('Perturbations Added')
plt.xlabel('Index')
plt.legend(loc='upper left')
plt.savefig('sizes_plot.png')
