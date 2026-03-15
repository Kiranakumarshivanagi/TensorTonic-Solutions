import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    """

    x = np.array(x, dtype=float)

    return 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))