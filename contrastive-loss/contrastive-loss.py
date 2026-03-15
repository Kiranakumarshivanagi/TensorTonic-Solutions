import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    """
    Compute Contrastive Loss for Siamese networks.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    y = np.array(y, dtype=float)

    # ensure batch dimension
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    # Euclidean distance
    d = np.linalg.norm(a - b, axis=1)

    # loss per sample
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2

    if reduction == "sum":
        return float(np.sum(loss))
    else:
        return float(np.mean(loss))