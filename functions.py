import numpy as np


def add_noise(xs, function, var = 9):
    return function(xs) + np.random.randn(len(xs))/var