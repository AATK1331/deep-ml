import numpy as np


def transition_matrix(A):
    # TODO: rows divided by out-degree; dangling rows uniform 1/n
    n=A.shape[0]
    degree=np.sum(A,axis=1,keepdims=True)
    p=np.where(degree>0,A/degree,1.0/n)
    return p


def stationary_distribution(A):
    # TODO: d_i / 2m for an undirected graph
    d=np.sum(A,axis=1)
    m=np.sum(A)/2.0

    return d/(2*m)


def pagerank(A, alpha=0.85, tol=1e-10, max_iter=1000):
    # TODO: r <- alpha P^T r + (1 - alpha)/n from uniform until L1 change < tol; return (r, n_iter)
    n=A.shape[0]
    P=transition_matrix(A)
    r=np.ones(n)/n
    for n_iter in range(max_iter):
        r_new=alpha*P.T@r+(1-alpha)/n
        delta=np.sum(np.abs(r_new-r))
        if delta<tol:
            return r_new,n_iter+1
        r=r_new
        
    return r,n_iter


def personalized_pagerank(A, seed, alpha=0.85, tol=1e-10, max_iter=1000):
    n=A.shape[0]
    P=transition_matrix(A)
    r=np.ones(n)/n
    e_seed=np.zeros(n)
    e_seed[seed]=1.0
    for n_iter in range(max_iter):
        r_new=alpha*P.T@r+(1-alpha)*e_seed
        delta=np.sum(np.abs(r_new-r))
        if delta<tol:
            return r_new,n_iter+1
        r=r_new
    return r,n_iter


