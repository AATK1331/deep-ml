import numpy as np


def qk_circuit(W_Q, W_K):
    return W_Q@W_K.T


def ov_circuit(W_V, W_O):
    
    return W_V@W_O

def circuit_rank(M, tol=1e-8):

    return np.linalg.matrix_rank(M,tol=tol)


def copying_score(W_OV):
    """Positive share of the real eigenvalue mass, rounded to 4 decimals."""
    eigvals=np.linalg.eigvals(W_OV)
    re=np.real(eigvals)
    numerator=np.sum(np.maximum(re,0.0))
    denominator=np.sum(np.abs(re))
    if denominator<1e-12:
        return 0.0
    score=numerator/denominator
    return np.round(score,4)

def full_ov_circuit(W_E, W_OV, W_U):
    """(W_E @ W_OV @ W_U, fraction of rows with the argmax on the diagonal)."""
    mat=W_E@W_OV@W_U
    n=mat.shape[0]
    cnt=0
    for i in range(n):
        if np.argmax(mat[i])==i:
            cnt+=1
    frac=round(cnt/n,4)
    return mat,frac

