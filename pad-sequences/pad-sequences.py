import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    
    # If empty input
    if len(seqs) == 0:
        return np.zeros((0,0), dtype=int)
    
    # Determine max length
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    
    N = len(seqs)
    
    # Create output array filled with pad_value
    padded = np.full((N, max_len), pad_value, dtype=int)
    
    # Fill values
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]  # truncate if longer
        padded[i, :len(seq)] = seq
    
    return padded