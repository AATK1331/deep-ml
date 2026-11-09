import numpy as np

def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)
        
    Returns:
        Output feature matrix of shape (N, F_out)
    """
    n=A.shape[0]
    A_tilde=A+np.eye(n)
    D=np.sum(A_tilde,axis=1)
    D_inv_sqrt=1.0/np.sqrt(D+1e-9)
    A_hat=np.diag(D_inv_sqrt)@A_tilde@np.diag(D_inv_sqrt)

    H=A_hat@X@W 
    return np.maximum(0,H)


    