import numpy as np
from ase.calculators.test import numeric_force
from ase.calculators.calculator import Calculator, all_changes
from mlvib.kernels import RBF

class Gaussian_process:
    def __init__(self, X, y, kernel, alpha=10**(-10)):
        """
        Gaussian Process for regression: 

        Parameters:
        kernel: func
            Function defining the kernel. Needs to be vectorized accepting 2-d arrays
        X: np.array (n, d)
            Data matrix for intial points.
        y: np.array (n, 1)
            Results for initial points.

        
        Properties:
        n: int
            Number of data points.
        d: int
            Dimension of feature vector.
        """

        # Settings:
        self.kernel = kernel
        self.alpha = alpha
        self.fitted = False

        
        # Data matrix:
        if X.ndim == 1:
            X = X.reshape(X.size, 1)

        self.X = X
        self.y = y
            
        # Dimensions:
        self.n, self.d = X.shape

        # Calculate the initial Kernel Matrix:
        self.K = self.kernel(self.X, self.X)
        
    def add_point(self, x, y):
        """
        Adds a data point:
        
        Parameters:
        -- x: np.array
           Feature vector of new data point.
        -- y: float
           Value for new data point.
        """
        if x.ndim == 1:
            x = x.reshape(1, x.size)
        if y.ndim == 1:
            y = y.reshape(y.size, 1)

        num_elems = x.shape[0]
        
        self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, y).reshape(-1, 1)
        self.n += num_elems
    
        # Update kernel matrix:
        new_values = self.kernel(x, self.X)
        self.K = np.vstack([self.K, new_values[:, 0:-num_elems]])
        self.K = np.hstack([self.K, new_values.T])
        
    def fit(self):
        """
        Invert matrix to find regression weights.
        """
        self.cov = np.linalg.inv(self.K+self.alpha*np.eye(self.n))
        self.w = self.cov@self.y
        self.fitted = True

    def predict(self, x, return_std=False):
        """
        Predict for the input point x
        """
        if x.ndim == 1:
            x.reshape(x.size, 1)

        K = self.kernel(x, self.X)
        out = (K @ self.w).reshape(len(x))
        
        if return_std == True:
            var = self.kernel.diag(x)
            var -= np.einsum("ij, ij->i", np.dot(K, self.cov), K)

            if (var < 0).any():
                var[var < 0] = 0
            
            return out, np.sqrt(var)
        else:
            return out

    def variance(self, x):
        if not hasattr(self, 'X'):
            return self.kernel.diag(x)
        else:
            K = self.kernel(x, self.X)
            var = self.kernel.diag(x)
            var -= np.einsum("ij, ij->i", np.dot(K, self.cov), K)

            if (var < 0).any():
                var[var < 0] = 0
        return np.sqrt(var)
        
    def score(self, X, y):
        prediction = self.predict(X).flatten()
        error = np.abs(prediction-y)
        return error


class GPCalculator(Calculator):

    def __init__(self, GP, descriptor, **kwargs):
        """
        ASE calculator-wrapper around Gaussian process class. 

        Inputs: 
        -- GP: Gaussian process instance.
        -- Descriptor: Descriptor class instance
        """

        Calculator.__init__(self, **kwargs)
        self.implemented_properties = ['energy', 'forces']
        
        self.GP = GP
        self.descriptor = descriptor
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Approximates the energy of the atoms object using Gaussian process regression
        """
        feature = self.descriptor(atoms).reshape(1, -1)
        self.results['energy'] = self.GP.predict(feature, return_std=False)

        # Calculate forces:
        if 'forces' in properties:

            self.results['forces'] = np.zeros((len(atoms), 3))
            # d = 0.001                                                                               
            # forces = np.zeros((len(atoms), 3))
            # for ii in range(len(atoms)):
            #    forces[ii] = [numeric_force(atoms, ii, i, d) for i in range(3)]
            # self.results['forces'] = forces

            # d = 0.001
            # features = np.zeros((6*len(atoms), feature.size))
            # c = 0
            # for a in range(len(atoms)):
            #     p0 = atoms.get_positions()
            #     for i in [0, 1, 2]:
            #         p = p0.copy()
            #         p[a, i] += d
            #         atoms.set_positions(p)
            #         features[c, :] = self.descriptor(atoms)
            #         p[a, i] -= 2*d
            #         atoms.set_positions(p)
            #         features[c+1, :] = self.descriptor(atoms)
            #         atoms.set_positions(p0)
            #         c += 2
            # E = self.GP.predict(features, return_std=False)
            # forces = np.zeros((len(atoms), 3))
            # c = 0
            # for i in range(len(atoms)):
            #     for dire in [0, 1, 2]:
            #         forces[i, dire] = (E[c+1]-E[c])/(2 * d)
            #         c += 2
            # self.results['forces'] = forces


            

        
        
    




