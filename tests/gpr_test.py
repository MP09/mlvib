import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from mlvib.kernels import RBF as myRBF
from mlvib.gaussian_process import Gaussian_process

import matplotlib.pyplot as plt

def compare_kernels(kernel1, kernel2):

    data = np.arange(9).reshape(3, 3)
    ref = np.array([1, 1, 1]).reshape(1, 3)


    ans1 = kernel1(data, ref)
    
    ans2 = kernel2(data, ref)
    
    print(np.allclose(ans1, ans2))
    

def GPR_regression():

    func = lambda x: x*np.sin(x)
    
    train_size = 10
    train_x = np.linspace(0, 10, train_size).reshape(-1, 1)
    train_y = func(train_x)

    # Scikit learn:
    gp = GaussianProcessRegressor(kernel=RBF())
    gp.fit(train_x, train_y)
    test_size = 100
    test_x = np.linspace(0, 10, test_size).reshape(-1, 1)
    test_y, sigma = gp.predict(test_x, return_std=True)

    plt.plot(train_x, train_y, 'o')
    plt.plot(test_x, func(test_x))
    plt.plot(test_x, test_y)
    plt.fill(np.concatenate([test_x, test_x[::-1]]),
         np.concatenate([test_y - 1.9600 * sigma,
                        (test_y + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

    plt.show()

def GPR_test():
    np.random.seed(1)


    def f(x):
        """The function to predict."""
        return x * np.sin(x)

    # ----------------------------------------------------------------------
    
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y = f(X).ravel()
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    kernel = RBF(1)
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)
    y_pred, sigma = gp.predict(x, return_std=True)
    plt.figure()
    plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    
    # -----------------------------------------------------------------------
    gp = Gaussian_process(X, y, kernel=myRBF)
    gp.fit()
    y_pred, sigma = gp.predict(x)
    plt.plot(x, y_pred, '--', color='black')
    # -----------------------------------------------------------------------


    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()

def variance_test():

    def f(x):
        """The function to predict."""
        return x * np.sin(x)

    # ----------------------------------------------------------------------
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y = f(X).ravel()

    kernel = RBF(1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0)
    gp.normalize_y = False
    gp.fit(X, y)
    x = np.atleast_2d(np.linspace(0, 10, 2)).T
    _, sigma = gp.predict(x, return_std=True)
    
    print(sigma)
    
    
    # ---------------------------------------------------------------------
    gp = Gaussian_process(X, y, kernel=kernel, alpha=0)
    gp.fit()
    y_pred, sigma = gp.predict(x)


    print(sigma)

def test():
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SKRBF
    from sklearn.kernel_ridge import KernelRidge
    
    np.random.seed(1)
    
    func = lambda x: x**2*np.sin(x)
    
    x = np.atleast_2d([1, 3, 5, 6, 7, 8]).T
    y = func(x)

    alpha = 10**(-5)
    
    GP = Gaussian_process(x, y, SKRBF(1), alpha=alpha)
    GP.fit()

    #plt.plot(x, y, 'o',  color='red', markersize=5)


    test_size = 100

    px = np.linspace(0, 10, test_size).reshape(test_size, 1)
    
    py, std = GP.predict(px)

    plt.plot(px, func(px), '-',  color='purple')
    plt.plot(px, py, '-', color='black')


    # Kernel Ridge comparison:
    from sklearn.kernel_ridge import KernelRidge
    clf = KernelRidge(alpha=alpha, kernel=SKRBF(1))
    clf.fit(x, y)

    py = clf.predict(px)
    plt.plot(px, py, '--',  color='red')
    
    gp = GaussianProcessRegressor(kernel=SKRBF(1), alpha=alpha)
    gp.fit(x, y)
    py, std = gp.predict(px, return_std = True)
    py = py.reshape(len(py))
    plt.plot(px, py, '--', color='lightgreen')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    # Comparison of Scikit learn RBF and my RBF:
    #compare_kernels(RBF(), myRBF)

    GPR_test()
    #variance_test()
    # 
