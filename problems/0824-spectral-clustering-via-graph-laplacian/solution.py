import numpy as np

def spectral_cluster(W, k):
    W=np.array(W,dtype=float)
    n=W.shape[0]
    W=np.array(W,dtype=float)
    if k==1:
        return [0]*n

    I=np.eye(n)
    d=np.sum(W,axis=1)
    d_inv_sqrt=np.where(d>0,1.0/np.sqrt(d),0)
    D_inv_sqrt=np.diag(d_inv_sqrt)
    L_sym=I-D_inv_sqrt@W@D_inv_sqrt
    

    eigenvalues,eigenvectors=np.linalg.eigh(L_sym)
    U=eigenvectors[:,:k]
    
    row_norms=np.linalg.norm(U,axis=1)
    T=np.zeros_like(U)
    for i in range(n):
        if row_norms[i]>0:
             T[i]=U[i]/row_norms[i]


    if k==0:
        return [0]*n
    centers=[T[0]]
    for _ in range(1,k):
        distances=[]
        for row in T:
            min_dist=min(np.linalg.norm(row-c)for c in centers)
            distances.append(min_dist)
        next_center_idx=np.argmax(distances)
        centers.append(T[next_center_idx])
    centers=np.array(centers)

    labels=np.zeros(n,dtype=int) 
    for _ in range(100):
        new_labels=np.zeros(n,dtype=int)
        for i in range(n):
             dists=np.linalg.norm(centers-T[i],axis=1)
             new_labels[i]=np.argmin(dists)
        if np.array_equal(new_labels,labels):
            break
        labels=new_labels
        for j in range(k):
            cluster_points=T[labels==j]
            if len(cluster_points)>0:
                centers[j]=np.mean(cluster_points,axis=0)
    mapping={}
    current_label=0
    final_labels=[]

    for label in labels:
        if label not in mapping:
            mapping[label]=current_label
            current_label+=1
        final_labels.append(mapping[label])

    return final_labels

