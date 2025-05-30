import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading data
train = pd.read_csv("trainst4.csv")
test = pd.read_csv("testst4.csv")

# adding labels and training
def encode_labels(y):
    classes = sorted(y.unique())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    inverse = {v: k for k, v in mapping.items()}
    return y.map(mapping), mapping, inverse

y_train, label_map, inv_label_map = encode_labels(train['color'])
y_test = test['color'].map(label_map)

X_train = train[['x', 'y']].to_numpy()
X_test = test[['x', 'y']].to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# training logistic regression from scratch
def train(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for epoch in range(epochs):
        z = np.dot(X, w) + b
        a = sigmoid(z)
        dz = a - y
        dw = np.dot(X.T, dz) / m
        db = np.sum(dz) / m

        w -= lr * dw
        b -= lr * db

    return w, b

# predict function
def predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    return (probs >= 0.5).astype(int)

# training model
w, b = train(X_train, y_train)

# prediction
y_pred = predict(X_test, w, b)

# final results and display
accuracy = np.mean(y_pred == y_test) * 100
print("Accuracy:", accuracy, "%")

plt.figure(figsize=(6, 6))
colors = [inv_label_map[val] for val in y_pred]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, cmap='bwr', s=1)
plt.title('Predicted Colors using Logistic Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
