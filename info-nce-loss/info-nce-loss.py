import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """

    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)

    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # Numerical stability (subtract row-wise max)
    S_max = np.max(S, axis=1, keepdims=True)
    S_exp = np.exp(S - S_max)

    # Softmax denominator
    denom = np.sum(S_exp, axis=1)

    # Numerator (positive pairs are diagonal)
    num = np.diag(S_exp)

    # InfoNCE loss
    loss = -np.mean(np.log(num / denom))

    return float(loss)