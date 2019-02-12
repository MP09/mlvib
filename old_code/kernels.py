
import numpy as np
from scipy.spatial.distance import euclidean, cdist

class RBF:

    def __init__(self, length_scale=1, metric='euclidean'):

        self.length_scale = length_scale
        self.metric = metric
        
    def __call__(self, X1, X2):
        return np.exp(-cdist(X1, X2, metric=self.metric)**2/(2*self.length_scale**2))
    
    def diag(self, x):
        return np.ones(x.shape[0])

class Dot:

    def __init__(self, bias=0):
        self.bias = bias

    def __call__(self, X1, X2):
        return X1 @ X2.T + self.bias

    def diag(self, x):
        return np.sum(x*x, axis=1) + self.bias
        
class RQ:

    def __init__(self, length_scale=1, alpha=1, metric='euclidean'):

        self.length_scale = 1
        self.alpha = 1
        self.metric = metric

    def __call__(self, X1, X2):
        return (1 + cdist(X1, X2, metric=self.metric)**2/(2*self.alpha*self.length_scale**2))**(-self.alpha)

    def diag(self, x):
        return np.ones(x.shape[0])
    

if __name__ == '__main__':

    X = np.arange(9).reshape(3, 3)
    print(X)

    kernel = RQ(alpha=10)

    
    


    
    
