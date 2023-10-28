import numpy as np


def gaussian_func(x, mu, sigma, A):
    a = A /(np.sqrt(2*np.pi)*sigma)
    b = -0.5 * (x-mu)**2/sigma**2
    func = a*np.exp(b)
    return func

def gaussian_linear_func(x, A, mu, sigma, k, c):
    a =  A /(np.sqrt(2*np.pi)*sigma)
    b =  -0.5 * (x-mu)**2/sigma**2
    func =  a*np.exp(b) +k*x+c
    return func
