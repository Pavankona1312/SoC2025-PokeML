import numpy as np
def softmax(Z):
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    return exp_Z / sum_exp_Z


def y_cap(X,W):
    Z = -X@W #X=nxm and W=mxc
    return softmax(Z)

def loss_likelihood(X,y,W):
    loss = (np.trace((X@W)@y.T))
    Z = - X @ W
    tmp = np.sum(np.log(np.sum(np.exp(Z), axis=1)))
    return (loss+tmp)/X.shape[0]

def gradient(X,y,W,lamda):
    n = X.shape[0]
    gre = ((X.T@(y-y_cap(X,W)))/n) + 2*lamda*W
    return gre

def gradient_des(X,y,W_in,lamda,alpha,iter,loss_needed=False):
    W=W_in
    loss=[]
    for _ in range(iter):
        W=W-alpha*gradient(X,y,W,lamda)
        if(loss_needed):
            loss.append(loss_likelihood(X,y,W))
    if(loss_needed):
        l = np.array(loss)
        return W,l
    else:
        return W