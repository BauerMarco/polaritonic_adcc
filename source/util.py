from libadcc import init_1d_tensor
from adcc import memory_pool
import numpy as np

def set_lt_scalar(scalar):
    """
    Helper function to initialize and set a libtensor
    of dimensionality and length one
    """
    ret = init_1d_tensor(1, memory_pool)
    ret.set_from_ndarray(np.array([scalar]))
    return ret

