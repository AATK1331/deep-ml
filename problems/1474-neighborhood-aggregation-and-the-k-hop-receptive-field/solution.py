import numpy as np
from collections import deque
def mean_aggregate(A, X):
    # TODO: row i = mean of X over the neighbours of i; zero row when no neighbours
    deg=np.sum(A,axis=1,keepdims=True)
    sum_neigh=(A@X).astype(float)
    out=np.where(deg>0,sum_neigh/deg,0.0)
    return out


def propagate(A, X, k, include_self=True):
    # TODO: k rounds of mean_aggregate, on A + I when include_sel
    n=A.shape[0]
    if include_self:
        A_new=A+np.eye(n)
    else:
        A_new=A

    for _ in range(k):
        X=mean_aggregate(A_new,X)
    return X


def k_hop_neighborhood(A, node, k):
    # TODO: sorted list of nodes within k hops, including node
    visited={node}
    current={node}
    for _ in range(k):
        next_set=set()
        for u in current:
            neighbors=np.where(A[u]==1)[0]
            next_set.update([int(v) for v in neighbors])
        current=next_set-visited
        visited=visited|next_set
    return sorted(visited)
#def k_hop_neighborhood(A, start, k):
#    n = A.shape[0]
#    dist = np.full(n, -1)
#    q = deque([start])
#    dist[start] = 0
#    while q:
#        u = q.popleft()
#        for v in range(n):
#            if A[u, v] == 1 and dist[v] == -1:
#                dist[v] = dist[u] + 1
#                q.append(v)
#    nodes = [i for i in range(n) if dist[i] == k]
#    return nodes

def hop_counts(A, node, K):
    res=[]
    for t  in range(0,K+1):
        neigh=len(k_hop_neighborhood(A,node,t))
        res.append(neigh)
    return res


