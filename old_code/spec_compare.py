import numpy as np

def gaussian(x, z, sigma):
    return np.exp(-(x-z)**2/sigma**2)

def freq_to_spec(w, x, sigma=0.01):
    """
    Converts array of inputs frequencies to continous spectrum. 

    Arguments:
    -- w: np.array of frequencies
    """
    return np.sum(gaussian(x, w[:, np.newaxis].real, sigma), axis=0)

def difference_integral(w1, w2, sigma=0.01):
    """
    Calculates the integral of the difference the spectrums s1-s2. 
    """
    max1 = np.max(w1)
    max2 = np.max(w2)
    
    x = np.linspace(0, np.max([w1, w2]), 2000)
    
    s1 = freq_to_spec(w1, x, sigma)
    s2 = freq_to_spec(w2, x, sigma)

    S = (s1-s2)**2

    return np.trapz(S, x)
