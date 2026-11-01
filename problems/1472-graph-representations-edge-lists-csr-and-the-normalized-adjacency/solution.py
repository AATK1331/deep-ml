import numpy as np


def edge_list_to_adjacency(edges, n, undirected=True):
    # TODO: (n, n) float matrix with a 1.0 for every edge (both directions if undirected)
    A=np.zeros((n,n),dtype=np.float32)
    for  u,v in edges:
        A[u,v]=1.0
        if undirected:
            A[v,u]=1.0
    return  A


def degrees(A):
    A=np.array(A)
    result1=np.sum(A,axis=1)
    return result1


def adjacency_to_csr(A):
    # TODO: return (indptr, indices) with neighbours of node i at indices[indptr[i]:indptr[i+1]]
    A=np.array(A)
    m,n=A.shape
    indptr=[0]
    indices=[]
    for i in range(m):
        neighbors=np.where(A[i]>0)[0]
        neighbors=sorted(neighbors)
        indices.extend(neighbors)
        indptr.append(len(indices))
    indices=np.array(indices)
    indptr=np.array((indptr))
    return (indptr,indices)

def normalized_adjacency(A, add_self_loops=True, mode="sym"):
    # TODO: D^-1/2 A D^-1/2 or D^-1 A after optional self-loops; zero-degree rows stay zero
    A=np.array(A)
    m,n=A.shape
    I=np.eye(n)
    
    if add_self_loops:
        A_tilde=A+I
    else:
        A_tilde=A
    d=degrees(A_tilde)
    d_mask=(d>0)
    d_safe=np.where(d_mask,d,1.0)
    
    if mode=="sym":
        d_inv_sqrt=np.where(d_mask,1/np.sqrt(d_safe),0.0)
        result=d_inv_sqrt[:,None]*A_tilde*d_inv_sqrt[None,:]
    elif mode=="rw":
        d_inv=np.where(d_mask,1/d_safe,0.0)
        result=d_inv[:,None]*A_tilde
    else:
        raise ValueError("mode must be 'sym' or 'rw'")
    return result
def num_walks(A, k):
    A=np.array(A)
    n=A.shape[0]
    result=np.eye(n)
    for _ in range(k):
        result=result@A
    return result
