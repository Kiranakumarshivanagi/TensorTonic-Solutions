import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """

    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # cosine similarity
    cos_sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    # loss based on label
    if label == 1:
        loss = 1 - cos_sim
    else:  # label == -1
        loss = max(0.0, cos_sim - margin)

    return float(loss)
    