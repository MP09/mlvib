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
        


if __name__ == '__main__':

    print('Kage')
