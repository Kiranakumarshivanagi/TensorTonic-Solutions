import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # focal loss formula
    loss = -( (1 - p)**gamma * y * np.log(p) +
              (p**gamma) * (1 - y) * np.log(1 - p) )

    return float(np.mean(loss))