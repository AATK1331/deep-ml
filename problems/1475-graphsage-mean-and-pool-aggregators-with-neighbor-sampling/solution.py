import numpy as np


def sample_neighbors(neighbors, num_samples, rng):
    # TODO: all neighbours (sorted) if few enough, else rng.choice without replacement, 
    
    if num_samples is None or len(neighbors)<=num_samples:
        return sorted(neighbors)
    else:
        sample=rng.choice(neighbors,size=num_samples,replace=False)
        return sorted(sample.tolist())


def sampled_neighbor_lists(adj_list, num_samples, rng):
    result=[]
    for neigh in adj_list:
        sampled=sample_neighbors(neigh,num_samples,rng)
        result.append(sampled)
    return result

def mean_aggregator(X, neighbor_lists):
    # TODO: mean of neighbour rows, zeros when empty
    out=[]
    feat_dim=X.shape[1]
    for neigh_ids in neighbor_lists:
        if len(neigh_ids)==0:
            agg=np.zeros(feat_dim)
        else:
            agg=np.mean(X[neigh_ids],axis=0)
        out.append(agg)
    return np.stack(out,axis=0)

def pool_aggregator(X, neighbor_lists, W_pool, b_pool):
    # TODO: elementwise max over neighbours of relu(X[j] @ W_pool + b_pool), zeros when empty
    out=[] 
    feat_dim=W_pool.shape[1]
    for neigh_ids in neighbor_lists:   
        if len(neigh_ids)==0:
            agg=np.zeros(feat_dim)
        else:
            neigh_feat=X[neigh_ids] 
            transformed=np.maximum(0,neigh_feat@W_pool+b_pool)
            agg=np.max(transformed,axis=0)
        out.append(agg)
    return np.stack(out,axis=0)
    


def sage_layer(X, neighbor_lists, W_self, W_neigh, aggregator="mean", W_pool=None, b_pool=None, normalize=True):
    # TODO: relu(X @ W_self + N @ W_neigh), then L2-normalize rows (zero rows stay zero)
    if aggregator=="mean":
        N=mean_aggregator(X,neighbor_lists)
    elif aggregator=="pool":
        N=pool_aggregator(X,neighbor_lists,W_pool,b_pool)
    else:
         raise ValueError("aggregator must be 'mean' or 'pool'")
    H=np.maximum(0,X@W_self+N@W_neigh)
    
    if normalize:
        row_norm=np.linalg.norm(H,axis=1,keepdims=True)
        mask=row_norm>1e-12
        H=np.where(mask,H/row_norm,H)
    return H


