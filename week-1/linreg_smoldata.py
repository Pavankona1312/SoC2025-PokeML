import numpy as np

with open('smoldata.csv','r') as data:
    data1 = data.read().splitlines()
    data1.pop(0)
    data1 = [i.split(',')[1:] for i in data1]
    data1 = np.array(data1,dtype=float)
X_ = data1[:,0:-2]
y1 = data1[:,-2]
y2 = data1[:,-1]

X_ = (X_ - X_.mean(axis=0))/X_.std(axis=0)
X = np.hstack((np.ones((X_.shape[0],1)),X_))
W = np.zeros((X.shape[1],))

def gradient(X,y,W):
    gre=(X.T)@(X@W-y)
    n=X.shape[0]
    return gre/n

def gradient_descent(X,y,W_in,alpha,iter):
    for _ in range(iter):
        W_in-=alpha*gradient(X,y,W_in)
    return W_in 

def r_squared(y,y_cap):
    num = np.sum((y-y_cap)*(y-y_cap))
    den = np.sum((y-np.mean(y))*((y-np.mean(y))))
    return 1-(num/den)

W1 = gradient_descent(X,y1,W,0.001,10000)
y1_cap = X@W1
print('R2 for mangoes:',r_squared(y1,y1_cap))

W2=gradient_descent(X,y2,W,0.001,10000)
y2_cap = X@W2
print('R2 for mangoes:',r_squared(y2,y2_cap))




