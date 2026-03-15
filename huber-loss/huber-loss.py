import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute mean Huber Loss.
    """

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # error
    e = y_true - y_pred
    abs_e = np.abs(e)

    # piecewise loss
    loss = np.where(abs_e <= delta,
                    0.5 * e**2,
                    delta * (abs_e - 0.5 * delta))

    return float(np.mean(loss))