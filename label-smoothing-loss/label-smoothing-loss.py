import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    
    p = np.array(predictions, dtype=float)
    K = len(p)

    # build smoothed target distribution
    q = np.full(K, epsilon / K)
    q[target] = (1 - epsilon) + (epsilon / K)

    # cross entropy loss
    loss = -np.sum(q * np.log(p))

    return float(loss)