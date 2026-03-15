import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    # positions (seq_len,1)
    pos = np.arange(seq_len)[:, np.newaxis]

    # dimension indices (1,d_model)
    dims = np.arange(d_model)[np.newaxis, :]

    # compute denominator term
    angle_rates = 1 / (base ** ((2 * (dims // 2)) / d_model))

    angles = pos * angle_rates

    # create positional encoding
    pe = np.zeros((seq_len, d_model), dtype=float)

    pe[:, 0::2] = np.sin(angles[:, 0::2])  # even columns
    pe[:, 1::2] = np.cos(angles[:, 1::2])  # odd columns

    return pe