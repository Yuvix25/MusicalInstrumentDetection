import math
import numpy as np
from scipy.interpolate import approximate_taylor_polynomial

def function_from_series(s):
    def func(x):
        if (type(x) == np.ndarray):
            return np.array([func(x1) for x1 in x])
        
        if (x < 0 or x > len(s)-1):
            return 0

        if (x == int(x)):
            return s[x]
        
        lower = math.floor(x)
        upper = math.ceil(x)
        return ((s[lower] - s[upper]) / (lower - upper)) * (x - lower) + s[lower]
    return func


def taylor_approx(func, length):
    return approximate_taylor_polynomial(func, 0, 35, length)



def timeline_fft(sr, envelope, chunk_sizes = 0.02):
    chunks = np.array_split(envelope, int(len(envelope)/(sr*chunk_sizes)))
    fft = np.array([np.fft.rfft(chunk) for chunk in chunks])
    return fft


def avg(x):
    return np.sum(x) / len(x)