import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)
        #self.relu2 = ReLULayer()


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        for param_key in params:
            param = params[param_key]
            param.grad = np.zeros_like(param.grad)
        
        loss, d_out = softmax_with_cross_entropy(self.layer2.forward(self.relu1.forward(self.layer1.forward(X))), y)
        self.layer1.backward(self.relu1.backward(self.layer2.backward(d_out)))

        for param_key in params:
            param = params[param_key]
            loss_reg, grad_regularization = l2_regularization(param.value, self.reg)
            loss += loss_reg
            param.grad += grad_regularization

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        return np.argmax(softmax(self.layer2.forward(self.relu1.forward(self.layer1.forward(X)))), axis=1)

    def params(self):
        result = {'W_layer1' : self.layer1.W, 'B_layer1' : self.layer1.B, 'W_layer2' : self.layer2.W, 'B_layer2' : self.layer2.B}
        return result
