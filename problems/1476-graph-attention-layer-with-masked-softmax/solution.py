import numpy as np


def leaky_relu(x, slope=0.2):
    
    return np.where(x>0,x,x*slope)


def attention_logits(H, a_src, a_dst, slope=0.2):
    # TODO: (n, n) matrix of leaky_relu(H[i] @ a_src + H[j] @ a_dst)
    n=H.shape[0]
    src_scores=H@a_src
    dst_scores=H@a_dst
    result=leaky_relu(src_scores[:,None]+dst_scores[None,:],slope)
    return result

def masked_softmax(E, mask):
    # TODO: row-wise softmax over masked entries only; zeros elsewhere
    E_masked=np.where(mask,E,-1e9)

    row_max=np.max(E_masked,axis=1,keepdims=True)
    E_stable=E_masked-row_max

    exp_E=np.exp(E_stable)
    exp_E=np.where(mask,exp_E,0.0)

    row_sum=np.sum(exp_E,axis=1,keepdims=True)
    return exp_E/row_sum

        


def gat_layer(A, X, W, a_src, a_dst, slope=0.2):
    # TODO: H = X @ W; mask = A + I > 0; alpha = masked softmax of the logits; return (alpha, alpha @ H)
    
    H=X@W
    n=A.shape[0]
    I=np.eye(n)
    mask=A+I>0
    logits=attention_logits(H,a_src,a_dst,slope)
    alpha=masked_softmax(logits,mask)
    return (alpha,alpha@H)

def multi_head_gat(A, X, Ws, a_srcs, a_dsts, concat=True, slope=0.2):
    outputs=[]

    for W,a_src,a_dst in zip(Ws,a_srcs,a_dsts):
        _,head_out=gat_layer(A,X,W,a_src,a_dst,slope)
        outputs.append(head_out)
    if concat:
       return np.concatenate(outputs,axis=1)
    else:
       return np.mean(outputs,axis=0)


