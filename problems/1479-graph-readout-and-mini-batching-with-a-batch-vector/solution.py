import numpy as np


def block_diagonal(mats):
    total_size=sum(m.shape[0] for  m in mats)
    result=np.zeros((total_size,total_size),dtype=mats[0].dtype)
    current_idx=0
    for m in mats:
        size=m.shape[0]
        result[current_idx:current_idx+size,current_idx:current_idx+size]=m

        current_idx+=size
    return result

def batch_graphs(adjs, feats):
    # TODO: return (A_block_diagonal, X_stacked, batch_vector)
    A=block_diagonal(adjs)
    X=np.vstack(feats)
    node_feats=[f.shape[0] for f in feats]
    batch=np.repeat(np.arange(len(adjs)),node_feats)

    return A,X,batch


def readout(H, batch, mode="sum"):
    # TODO: (num_graphs, d) with sum / mean / max per graph
    G=int(batch.max())+1
    d=H.shape[1]
    result=np.zeros((G,d),dtype=H.dtype)
    
    for g in range(G):
        node_feats=H[batch==g]
        if len(node_feats)>0:
            if mode=='sum':
               result[g]=np.sum(node_feats,axis=0)
            elif mode=='mean':
               result[g]=np.mean(node_feats,axis=0)
            elif mode=='max':
                result[g]=np.max(node_feats,axis=0)
        
    return result

def unbatch(H, batch):
    counts=np.bincount(batch)
    split_indices=np.cumsum(counts)[:-1]
    return np.split(H,split_indices)



