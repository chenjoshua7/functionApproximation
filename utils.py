import math
import numpy as np

def create_polynomials(xs, N = 10, A:float = 0.):
    xs = xs.reshape(len(xs), 1)
    xs_copy = xs.copy() - A
    for n in range(2, N + 1):
        xs = np.concatenate([xs, (xs_copy)**n/math.factorial(n)], axis = 1)
    return xs

def crazy_func(xs):
    return np.sqrt(np.exp(xs/10) * np.cos(xs) + 4)

def cosine_with_noise(xs):
    return np.cos(xs) + np.random.randn(len(xs))/9
