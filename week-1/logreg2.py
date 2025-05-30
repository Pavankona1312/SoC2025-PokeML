import numpy as np
import matplotlib.pyplot as plt
from logregfunctions import gradient_des,y_cap
import time

start = time.time()
with open('trainst2.csv','r') as f:
    dt1 = f.read().splitlines()
    dt1.pop(0)
    dt1 = [i.split(',') for i in dt1]
    data = np.array(dt1)

X = data[:,0:-1].astype(float)
X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
y = data[:,-1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

plt.plot(X[y_encoded==0,0], X[y_encoded==0,1], 'r+', X[y_encoded==1,0], X[y_encoded==1,1], 'b_', X[y_encoded==2,0], X[y_encoded==2,1], 'g+',X[y_encoded==3,0], X[y_encoded==3,1], 'm_', X[y_encoded==4,0], X[y_encoded==4,1], 'y+', X[y_encoded==5,0], X[y_encoded==5,1], 'k_')
plt.title('Plot of trainig data')
plt.show()

from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder()
y_onehotencoded = oe.fit_transform(y.reshape(-1,1)).toarray()

X_train = np.column_stack((np.ones((X.shape[0],1)),X))
y_train = y_onehotencoded
W = np.zeros((X_train.shape[1],y_train.shape[1]))

W_fin,loss_arr = gradient_des(X_train,y_train,W,0,0.3,1000,loss_needed=True)

y_res = y_cap(X_train,W_fin)
y_res = np.argmax(y_res,axis=1)

print(y_encoded)
print(y_res)
acc=abs(y_encoded-y_res)>0
a=0
for i in acc:
    if(i):
        a+=1
print('Accuracy of training data:',(1-a/y_res.shape[0])*100)
plt.plot(loss_arr,'r+')
plt.title('Loss_Likelihood vs Iterations')
plt.show()

with open('testst2.csv','r') as f:
    dt1 = f.read().splitlines()
    dt1.pop(0)
    dt1 = [i.split(',') for i in dt1]
    data = np.array(dt1) 

X = data[:,0:-1].astype(float)
X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
y = data[:,-1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

plt.plot(X[y_encoded==0,0], X[y_encoded==0,1], 'r+', X[y_encoded==1,0], X[y_encoded==1,1], 'b_', X[y_encoded==2,0], X[y_encoded==2,1], 'g+',X[y_encoded==3,0], X[y_encoded==3,1], 'm_', X[y_encoded==4,0], X[y_encoded==4,1], 'y+', X[y_encoded==5,0], X[y_encoded==5,1], 'k_')

X_test = np.column_stack((np.ones((X.shape[0],1)),X))

y_test_res=y_cap(X_test,W_fin)
y_test_res = np.argmax(y_test_res,axis=1)

print(y_encoded)
print(y_test_res)

acc=abs(y_encoded-y_test_res)>0
a=0
for i in acc:
    if(i):
        a+=1
print('Accuracy of testing data:',(1-a/y_test_res.shape[0])*100)

print(f'task is done in {time.time()-start}sec')