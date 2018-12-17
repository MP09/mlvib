import matplotlib.pyplot as plt
from mlvib.plotters import plot_spectrum
import numpy as np


x = np.linspace(0, 0.45, 1000)
fig, ax = plt.subplots()
for j in [10, 50, 100, 200, 323]:

    w = np.load('data/w{}.npy'.format(j))

    plot_spectrum(w, sigma=0.01, x=x, fig=fig, ax=ax, label=j)


w = np.load('w_EMT.npy')

plot_spectrum(w, sigma=0.01, fig=fig, ax=ax, label='EMT', color='red', linestyle='--')
plt.xlim([0, max(x)])
plt.legend()
plt.tight_layout()
plt.show()


wEMT = np.load('w_EMT.npy')
w = np.load('data/w323.npy')

for w1, w2 in zip(w, wEMT):
    print('GP: {}, True: {}'.format(w1.real, w2.real))
