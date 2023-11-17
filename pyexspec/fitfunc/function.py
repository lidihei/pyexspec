import numpy as np

def gaussian_func(x, A, mu, sigma):
    '''
    y = A /(sqrt(2*np.pi)*sigma) exp(-0.5 * (x-mu)**2/sigma**2)
    parameter:
    x [array of float]
    A, mu, simga [float]
    returns:
    --------
    y [array or float]
    '''
    a = A /(np.sqrt(2*np.pi)*np.abs(sigma))
    b = -0.5 * (x-mu)**2/sigma**2
    func = a*np.exp(b)
    return y

def gaussian_linear_func(x, A, mu, sigma, k, c):
    '''
    y = A /(sqrt(2*np.pi)*sigma) exp(-0.5 * (x-mu)**2/sigma**2) + kx+c
    parameter:
    x [array of float]
    A, mu, simga, k, c [float]
    returns:
    --------
    y [array or float]
    '''
    a =  A /(np.sqrt(2*np.pi)*np.abs(sigma))
    b =  -0.5 * (x-mu)**2/sigma**2
    y =  a*np.exp(b) +k*x+c
    return y
