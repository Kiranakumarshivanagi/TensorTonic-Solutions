import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores
    reduction: "mean" or "sum"
    """

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # compute hinge loss for each sample
    loss = np.maximum(0, margin - y_true * y_score)

    if reduction == "sum":
        return float(np.sum(loss))
    else:  # default mean
        return float(np.mean(loss))