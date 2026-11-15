import numpy as np

def aggregate_multiset(values, mode):
    # TODO: sum / mean / max of the list; 0.0 when empty
    if len(values)==0:
        return 0.0
    if mode=='sum':
        return float(np.sum(values))
    elif mode=='mean':
        return float(np.mean(values))
    elif mode=='max':
        return float(np.max(values))
    


def count_distinct(multisets, mode):
    agg_values=[]
    for ms in multisets:
        val=aggregate_multiset(ms,mode)
        is_district=True
        for existing in agg_values:
             if abs(val-existing)<=1e-9:
                is_district=False
                break
        if is_district:
            agg_values.append(val)
    return len(agg_values)
        

def gin_aggregate(A, X, eps=0.0):
    # TODO: (1 + eps) X + A X
    return (1+eps)*X+A@X


def mlp(H, W1, b1, W2, b2):
    
    return np.maximum(0,H@W1+b1)@W2+b2


def gin_layer(A, X, W1, b1, W2, b2, eps=0.0):
    
    return mlp(gin_aggregate(A,X,eps),W1,b1,W2,b2)


def gin_graph_embedding(A, X, params, eps=0.0):
    # TODO: apply the layers in sequence, sum the final node states
    for W1,b1,W2,b2 in params:
        X=gin_layer(A,X,W1,b1,W2,b2,eps)
    return np.sum(X,axis=0)
