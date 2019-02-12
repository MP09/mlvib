import numpy as np
import matplotlib.pyplot as plt
from mlvib.gaussian_process import Gaussian_process
from mlvib.kernels import RBF

# Create some data:
xlim = np.array([-5, 5])
prior = 100
x = np.random.uniform(xlim[0], xlim[1], 50)
y = x**2 + 10

kernel = RBF()
GP = Gaussian_process(x, y, kernel, prior=prior)
GP.fit()

x_ = np.linspace(*xlim, 200).reshape(-1, 1)
y_ = GP.predict(x_)

plt.plot(x, y, 'o', color='red')
plt.plot(x_, y_, '-', color='black')
plt.tight_layout()
plt.show()
