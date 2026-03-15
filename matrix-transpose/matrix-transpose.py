import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    
    A = np.array(A)
    
    N, M = A.shape
    
    # Create empty matrix with swapped shape
    AT = np.zeros((M, N), dtype=A.dtype)
    
    # Fill values
    for i in range(N):
        for j in range(M):
            AT[j][i] = A[i][j]
    
    return AT