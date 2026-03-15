import numpy as np

def dropout(x, p=0.5, seed=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """

    x = np.array(x, dtype=float)

    # Create RNG using seed
    if seed is None:
        rand = np.random.random(x.shape)
    else:
        rng = np.random.default_rng(seed)
        rand = rng.random(x.shape)

    # Keep mask
    keep = rand < (1 - p)

    # Scaling
    scale = 1.0 / (1.0 - p)

    dropout_pattern = keep.astype(float) * scale
    output = x * dropout_pattern

    return output, dropout_pattern