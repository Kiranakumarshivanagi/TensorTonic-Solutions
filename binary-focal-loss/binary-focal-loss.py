import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """

    p = np.array(predictions, dtype=float)
    y = np.array(targets, dtype=float)

    # probability of the true class
    p_t = np.where(y == 1, p, 1 - p)

    # focal loss
    loss = -alpha * ((1 - p_t) ** gamma) * np.log(p_t)

    return float(np.mean(loss))