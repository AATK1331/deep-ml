import numpy as np


def scatter_sum(M, index, n):
    # TODO: rows of M summed into their index; use np.add.at
    m,d=M.shape
    result=np.zeros((n,d),dtype=M.dtype)
    np.add.at(result,index,M)  
    return  result

def scatter_mean(M, index, n):
    sums=scatter_sum(M,index,n)
    
    counts=np.bincount(index,minlength=n)
    counts[counts==0]=1.0

    return sums/counts[:,None]

def edge_messages(X, E, edge_index, W_node, W_edge, b):
    # TODO: relu(X[src] @ W_node + E @ W_edge + b), one row per edge
    
    src=edge_index[0]
    msg=X[src]@W_node+E@W_edge+b
    
    return np.maximum(0,msg)

def to_undirected(edge_index, E):
    # TODO: append reversed edges with the same features
    
    reverse_edges=edge_index[::-1,:]

    new_edge_index=np.concatenate([edge_index,reverse_edges],axis=1)
    new_E=np.concatenate([E,E],axis=0)

    return new_edge_index,new_E


def mpnn_layer(X, E, edge_index, W_node, W_edge, b, W_self, W_agg, aggregate="sum"):
    # TODO: messages -> scatter to destinations -> relu(X @ W_self + agg @ W_agg)
    n=X.shape[0]
    dst=edge_index[1]
    msg=edge_messages(X,E,edge_index,W_node,W_edge,b)
    
    if aggregate=='sum': 
        agg=scatter_sum(msg,dst,n)
    elif aggregate=='mean':
        agg=scatter_mean(msg,dst,n)

    out=X@W_self+agg@W_agg

    return np.maximum(0,out)
