import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NN:
    def __init__(self, input_dim, hidden_dim, activation_func=sigmoid, activation_derivative=sigmoid_derivative):
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.w1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.random.randn(1, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, 1)
        self.b2 = np.random.randn(1, 1)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.w1 + self.b1     # (N, hidden_dim)
        self.a1 = self.activation_func(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2  # (N, 1)
        self.yhat = sigmoid(self.z2)
        return self.yhat

    def backward(self, X, y, learning_rate):
        N = y.shape[0]
        y = y.reshape(-1, 1)

        # Output layer
        dz2 = self.yhat - y                      # (N, 1)
        dw2 = (self.a1.T @ dz2) / N              # (hidden_dim, 1)
        db2 = np.sum(dz2, axis=0, keepdims=True) / N

        # Hidden layer
        da1 = dz2 @ self.w2.T                    # (N, hidden_dim)
        dz1 = da1 * self.activation_derivative(self.z1)  # (N, hidden_dim)
        dw1 = (X.T @ dz1) / N                    # (input_dim, hidden_dim)
        db1 = np.sum(dz1, axis=0, keepdims=True) / N

        # Update weights and biases
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

            # Loss and accuracy
            eps = 1e-8  # avoid log(0)
            loss = np.mean(-y * np.log(self.yhat + eps) - (1 - y) * np.log(1 - self.yhat + eps))
            accuracy = np.mean((self.yhat > 0.5).reshape(-1,) == y)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            self.pred('pred_train.txt')

    def pred(self, file_name='pred.txt'):
        pred = self.yhat > 0.5
        with open(file_name, 'w') as f:
            for i in range(len(pred)):
                f.write(str(self.yhat[i][0]) + ' ' + str(int(pred[i])) + '\n')

# Main runner
if __name__ == "__main__":
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"

    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)

    X = data[:, :-1]
    y = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    input_dim = X.shape[1]
    hidden_dim = 4
    learning_rate = 0.05
    num_epochs = 100

    model = NN(input_dim, hidden_dim)
    model.train(X**2, y, learning_rate, num_epochs)  # train on squared features (for concentric circles)

    test_preds = model.forward(X_eval**2)
    model.pred('pred_eval.txt')

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Test accuracy: {test_accuracy:.4f}")
