import numpy as np

def sigmoid(x):
    """
    Implement sigmoid activation function.
    Works with scalars, lists, and numpy arrays.
    """
    
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    
    # Sigmoid computation
    result = 1 / (1 + np.exp(-x))
    
    return result