import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic

# Function be fit:
xlim = [0, 10]
num_points = 20
func = lambda x: np.sin(x)*x**2*np.exp(0.1*x)

rng = np.random.RandomState(0)
X = rng.uniform(xlim[0], xlim[1], num_points)[:, np.newaxis]
y = func(X)

# Kernel:
kernels = []

#RBF_W = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
#    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

#kernels.append(RBF_W)

RBF = 1.0 * RBF(length_scale=1, length_scale_bounds=(0.5, 10))

kernels.append(RBF)

# Matern kernels:
#for nu in [1/2, 3/2, 5/2]:
#    kernels.append(Matern(nu=nu))

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
#axes = axes.flatten()
axes = [axes]

for ax, kernel in zip(axes, kernels):
# Setting up GP

    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.00000001, n_restarts_optimizer=1)
    gp.fit(X, y)

    
    # Prediction:
    X_ = np.linspace(xlim[0], xlim[1], 100)
    y_mean, y_cov = gp.predict(X_[:, np.newaxis], return_cov=True)



    ax.plot(X, y, 'o', color='red')
    ax.plot(X_, y_mean, color='black')
    ax.plot(X_, func(X_), '--', color='red')

plt.tight_layout()    
plt.show()



