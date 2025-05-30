import pandas as pd
import numpy as np

# loading data 
train = pd.read_csv("trainst3.csv")
test = pd.read_csv("testst3.csv")

# data cleaning 
def parse_labels(labels):
    return labels.apply(lambda x: x.split(','))

y_train = parse_labels(train['color'])
y_test = parse_labels(test['color'])

# one-hot encoding
all_labels = sorted(set(l for sublist in y_train for l in sublist))
label_to_index = {label: i for i, label in enumerate(all_labels)}
index_to_label = {i: label for label, i in label_to_index.items()}

def encode(y):
    encoded = np.zeros((len(y), len(all_labels)))
    for i, labels in enumerate(y):
        for label in labels:
            encoded[i, label_to_index[label]] = 1
    return encoded

Y_train = encode(y_train)
Y_test = encode(y_test)

# conditions and training
X_train = train[['x', 'y']].to_numpy()
X_test = test[['x', 'y']].to_numpy()

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# training logistic regression from scratch
def train_logistic(X, Y, lr=0.1, epochs=1000):
    m, n = X.shape
    k = Y.shape[1]
    W = np.zeros((n, k))
    b = np.zeros((1, k))

    for epoch in range(epochs):
        Z = np.dot(X, W) + b
        A = sigmoid(Z)
        dZ = A - Y
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        W -= lr * dW
        b -= lr * db

    return W, b

# predict function
def predict(X, W, b):
    probs = sigmoid(np.dot(X, W) + b)
    return (probs >= 0.5).astype(int)

# training model
W, b = train_logistic(X_train, Y_train)

# prediction
Y_pred = predict(X_test, W, b)

# final result 
accuracy = np.mean(Y_pred == Y_test) * 100
print("Accuracy", accuracy, "%")
