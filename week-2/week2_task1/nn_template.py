import numpy as np

np.random.seed(42)

"""
Sigmoid activation applied at each node.
"""
def sigmoid(x):
    # cap the data to avoid overflow?
    # x[x>100] = 100
    # x[x<-100] = -100
    return 1/(1+np.exp(-x))

"""
Derivative of sigmoid activation applied at each node.
"""
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN:
    def __init__(self, input_dim, hidden_dim, activation_func = sigmoid, activation_derivative = sigmoid_derivative):
        """
        Parameters
        ----------
        input_dim : TYPE
            DESCRIPTION.
        hidden_dim : TYPE
            DESCRIPTION.
        activation_func : function, optional
            Any function that is to be used as activation function. The default is sigmoid.
        activation_derivative : function, optional
            The function to compute derivative of the activation function. The default is sigmoid_derivative.

        Returns
        -------
        None.

        """
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        # TODO: Initialize weights and biases for the hidden and output layers
        self.W_b_h = np.random.normal(loc=0, scale=1, size=(hidden_dim,input_dim+1))
        self.W_b_o  = np.random.normal(loc=0, scale=1, size=(1,hidden_dim+1))
        
    def forward(self, X):
        # Forward pass
        # TODO: Compute activations for all the nodes with the activation function applied
        # for the hidden nodes, and the sigmoid function applied for the output node
        Xp = np.column_stack((np.ones((X.shape[0],)),X))
        h_activations = self.activation_func((self.W_b_h)@(Xp.T))
        h_activations = h_activations.T
        hp = np.column_stack((np.ones((h_activations.shape[0],)),h_activations))
        output = sigmoid((self.W_b_o)@(hp.T))
        # TODO: Return: Output probabilities of shape (N, 1) where N is number of examples
        return output.reshape((X.shape[0],1))
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        # TODO: Compute gradients for the output layer after computing derivative of sigmoid-based binary cross-entropy loss
        tmp = self.forward(X)
        Xp = np.column_stack((np.ones((X.shape[0],)),X))
        h_a = self.activation_func((self.W_b_h)@(Xp.T))
        h_a = h_a.T
        h_a = np.column_stack((np.ones((h_a.shape[0],)),h_a))
        N=X.shape[0]
        gre_o = (h_a.T@(tmp-y.reshape((-1,1))))/N
        gre_o = gre_o.T
        # TODO: When computing the derivative of the cross-entropy loss, don't forget to divide the gradients by N (number of examples)  
        # TODO: Next, compute gradients for the hidden layer
        W_o = self.W_b_o[:, 1:]
        gre_h = np.zeros(self.W_b_h.shape)
        for i in range(N):
            y_cap = sigmoid((self.W_b_o)@(h_a[i]))
            tmp1 = np.array([-y[i]+y_cap])
            x = Xp[i].reshape((-1,1))
            f_dash = self.activation_derivative((self.W_b_h)@x)
            gre_h += np.diag(f_dash.flatten())@((W_o).T)@(tmp1)@(x.T)
        gre_h=gre_h/N
        # TODO: Update weights and biases for the output layer with learning_rate applied
        self.W_b_h -= learning_rate*gre_h
        # TODO: Update weights and biases for the hidden layer with learning_rate applied
        self.W_b_o -= learning_rate*gre_o

    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            self.forward(X)
            # Backpropagation and gradient descent weight updates
            self.backward(X, y, learning_rate)
            # TODO: self.yhat should be an N times 1 vector containing the final
            # sigmoid output probabilities for all N training instances 
            self.yhat = self.forward(X)
            # TODO: Compute and print the loss (uncomment the line below)
            loss = np.mean(-y*np.log(self.yhat) - (1-y)*np.log(1-self.yhat))
            # TODO: Compute the training accuracy (uncomment the line below)
            accuracy = np.mean((self.yhat > 0.5).reshape(-1,) == y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            self.pred('pred_train.txt')
            
    def pred(self,file_name='pred.txt'):
        pred = self.yhat > 0.5
        with open(file_name,'w') as f:
            for i in range(len(pred)):
                f.write(str(self.yhat[i]) + ' ' + str(int(pred[i])) + '\n')

# Example usage:
if __name__ == "__main__":
    # Read from data.csv 
    csv_file_path = "data_train.csv"
    eval_file_path = "data_eval.csv"
    
    data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=0)
    data_eval = np.genfromtxt(eval_file_path, delimiter=',', skip_header=0)
    # Separate the data into X (features) and y (target) arrays
    X = data[:, :-1]
    y = data[:, -1]
    X_eval = data_eval[:, :-1]
    y_eval = data_eval[:, -1]

    # Create and train the neural network
    input_dim = X.shape[1]
    hidden_dim = 4
    learning_rate = 0.05
    num_epochs = 100
    
    model = NN(input_dim, hidden_dim)
    model.train(X**2, y, learning_rate, num_epochs) #trained on concentric circle data 

    test_preds = model.forward(X_eval**2)
    model.pred('pred_eval.txt')

    test_accuracy = np.mean((test_preds > 0.5).reshape(-1,) == y_eval)
    print(f"Test accuracy: {test_accuracy:.4f}")
