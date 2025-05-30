import numpy as np
import matplotlib.pyplot as plt
import time
from logregfunctions import gradient_des,y_cap
from datasets import load_dataset

start=time.time()
ds = load_dataset("ylecun/mnist")
train_data = ds['train']
test_data = ds['test']
X_train, y_train = np.array(train_data['image']), np.array(train_data['label'])
X_test, y_test = np.array(test_data['image']), np.array(test_data['label'])

print(f"The dataset got imported in {time.time()-start}sec")
dt = np.reshape(X_train,(X_train.shape[0],-1))

from sklearn.preprocessing import OneHotEncoder
oe = OneHotEncoder()
y_train_onehot = oe.fit_transform(y_train.reshape(-1,1)).toarray()

X = np.column_stack((np.ones((dt.shape[0],1)),dt))
y = y_train_onehot
W = np.zeros((X.shape[1],y.shape[1]))

print('Working on weights')
W_fin = gradient_des(X,y,W,0.01,0.00001,500) #Change the no.of iterations and alpha here
#If you need the loss graph, Add loss_needed=True in the above function so you can get loss array as output too. Make sure to change W_fin to W_fin,loss.
#Also, It will take around an hour to get whole array.
y_res = y_cap(X,W_fin)


y_res = np.argmax(y_res,axis=1)
acc = abs(y_train - y_res)>0
a=0
for i in acc:
    if i :
        a+=1
print((1-a/y_train.shape[0])*100)


dt2 = np.reshape(X_test,(X_test.shape[0],-1))
x_test = np.column_stack((np.ones((dt2.shape[0],1)),dt2))
y_test_res = y_cap(x_test,W_fin)
y_test_res = np.argmax(y_test_res,axis=1)

print(y_test)
print(y_test_res)
acc = abs(y_test - y_test_res)>0
a=0
for i in acc:
    if i :
        a+=1
print((1-a/y_test.shape[0])*100)

print(f'Task is done in {time.time()-start}sec')