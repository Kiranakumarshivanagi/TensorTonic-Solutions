import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q)
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # add epsilon to avoid log(0)
    q = q + eps

    # compute only where p > 0
    mask = p > 0

    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return float(kl)