import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def gaussian(x, z, sigma):
    return np.exp(-(x-z)**2/sigma**2)

def plot_spectrum(w, sigma=0.001, x=None, fig=None, ax=None, **kwargs):

    w = w[w.real > 0].real
    
    # Data stuff:
    if x is None:
        x = np.linspace(0, max(w)*1.15, 1000)
    y = np.sum(gaussian(x, w[:, np.newaxis].real, sigma), axis=0)

    # Figure stuff:
    if fig is None:
        fig, ax = plt.subplots()
    
    ax.plot(x, y, **kwargs)
    
    return fig, ax

def feature_histogram(features, fig=None, ax=None):
    """
    Plots a histogram of the distance between features:

    Arguments:
    -- features: np.array (n, d)
    """

    dists = cdist(features, features)


    if fig is None:
        fig, ax = plt.subplots()
    
    ax.hist(dists)

    ax.set_xlabel('Feat. dist [?]')
    ax.set_ylabel('Counts')

    plt.tight_layout()

    return fig, ax

    
