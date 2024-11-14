import numpy as np

def sub2ind(siz, x):
    """Convert subscripts to a linear index for matrix sizes."""
    k = np.cumprod([1] + siz[:-1])
    return sum(x_i * k_i for x_i, k_i in zip(x, k))

