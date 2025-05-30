import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
with open('trainst1.csv','r') as f:
    dt1 = f.read().splitlines()
    dt1.pop(0)
    dt1 = [i.split(',') for i in dt1]
    data = np.array(dt1,dtype=float)

x0=[]
x1=[]
for i in data:
    if (i[2] == 0):
        x0.append(i[0:-1])
    else:
        x1.append(i[0:-1])
x1 = np.array(x1)
x0 = np.array(x0)
plt.scatter(x0[:,0],x0[:,1])
plt.scatter(x1[:,0],x1[:,1])
plt.show()

X_train = data[:,0:-1]
X_train = (X_train - np.mean(X_train,axis=0))/np.std(X_train,axis=0)
y_train = data[:,-1]
X_train = np.column_stack((X_train,X_train[:,0]**2))
X_train = np.column_stack((np.ones((X_train.shape[0],1)),X_train))

W = np.zeros((X_train.shape[1]))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradient(X,y,W):
    y_cap = X@W
    tmp = np.array([sigmoid(x) for x in y_cap])
    gre=X.T@(tmp-y)
    return gre

def gradient_descent(X,y,W_in,alpha,itr):
    W=W_in
    for i in range(itr):
        tmp = gradient(X,y,W)
        W -= alpha*tmp
    return W

W_fin = gradient_descent(X_train,y_train,W,alpha=0.001,itr=500)
y_cap = sigmoid(X_train@W_fin)
for i in range(y_cap.shape[0]):
    if(y_cap[i]>=0.5):
        y_cap[i] = 1
    else:
        y_cap[i] = 0
print(y_cap)
print(y_train)
acc = abs(y_train - y_cap)>0
a=0
for i in acc:
    if i :
        a+=1
print('Accuracy of training data:',(1-a/y_train.shape[0])*100)

with open('testst2.csv','r') as f:
        dt1 = f.read().splitlines()
        dt1.pop(0)
        dt1 = [i.split(',') for i in dt1]
        data = np.array(dt1,dtype=float)

X_test = data[:,0:-1]
X_test = (X_test - np.mean(X_test,axis=0))/np.std(X_test,axis=0)
y_test = data[:,-1]
X_test = np.column_stack((X_test,X_test[:,0]**2))
X_test = np.column_stack((np.ones((X_test.shape[0],1)),X_test))

y_cap_test = sigmoid(X_test@W_fin)

for i in range(y_cap_test.shape[0]):
    if(y_cap_test[i]>=0.5):
        y_cap_test[i] = 1
    else:
        y_cap_test[i] = 0
acc = abs(y_test - y_cap_test)>0
a=0
for i in acc:
    if i :
        a+=1
print('Accuracy of testng data:',(1-a/y_test.shape[0])*100)
print(f'Task is finished in {time.time()-start}sec')