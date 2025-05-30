import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading data 
train_df = pd.read_csv("trainst2.csv")
test_df = pd.read_csv("testst2.csv")

# inputs
X_train = train_df[['x', 'y']].to_numpy()
X_test = test_df[['x', 'y']].to_numpy()

# normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# add bias column
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# labels
y_train_raw = train_df['color']
y_test_raw = test_df['color']

classes = sorted(y_train_raw.unique())
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
index_to_class = {idx: cls for cls, idx in class_to_index.items()}
y_train = np.array([class_to_index[c] for c in y_train_raw])
y_test = np.array([class_to_index[c] for c in y_test_raw])

# one hot encoding
def one_hot(y, num_classes):
    m = y.shape[0]
    oh = np.zeros((m, num_classes))
    oh[np.arange(m), y] = 1
    return oh

Y_train = one_hot(y_train, len(classes))

# softmax
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)  # stability improvement
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# train softmax regression
def train_softmax(X, Y, lr=0.05, epochs=1000):
    m, n = X.shape
    k = Y.shape[1]
    W = np.zeros((n, k))
    for _ in range(epochs):
        Z = np.dot(X, W)
        A = softmax(Z)
        dW = np.dot(X.T, A - Y) / m
        W -= lr * dW
    return W

# prediction
def predict(X, W):
    probs = softmax(np.dot(X, W))
    return np.argmax(probs, axis=1)

# train the model
W = train_softmax(X_train, Y_train)

# make predictions
y_pred_idx = predict(X_test, W)
y_pred_labels = [index_to_class[i] for i in y_pred_idx]

# final results and display 
accuracy = np.mean(y_pred_idx == y_test) * 100
print("Accuracy:", accuracy, "%")

# visualize results
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_idx, cmap='tab10', s=20, edgecolors='k', alpha=0.7)
plt.title("Predicted Classes - Softmax Regression")
plt.xlabel("x (normalized)")
plt.ylabel("y (normalized)")
plt.grid(True)
plt.show()
