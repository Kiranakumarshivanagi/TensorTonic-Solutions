import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """

    y = np.array(y)

    # Handle empty node
    if y.size == 0:
        return 0.0

    # Count class occurrences
    _, counts = np.unique(y, return_counts=True)

    # Convert to probabilities
    p = counts / counts.sum()

    # Compute entropy (ignore p=0 automatically)
    entropy = -np.sum(p * np.log2(p))

    return float(entropy)