import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.conv1 = ConvolutionalLayer(input_shape[-1], conv1_channels, filter_size=3, padding=1)
        self.relu1 = ReLULayer()
        self.maxpool1 = MaxPoolingLayer(pool_size=2, stride=2)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size=3, padding=1)
        self.relu2 = ReLULayer()
        self.maxpool2 = MaxPoolingLayer(pool_size=2, stride=2)
        self.flattener = Flattener()
        self.fully_connected = FullyConnectedLayer(8*8*conv2_channels, n_output_classes)

    def forward_without_softmax(self, X):
        return self.fully_connected.forward(self.flattener.forward(self.maxpool2.forward(
            self.relu2.forward(self.conv2.forward(self.maxpool1.forward(self.relu1.forward(self.conv1.forward(X))))))))

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params = self.params()
        for param_key in params:
            param = params[param_key]
            param.grad = np.zeros_like(param.grad)

        loss, d_out = softmax_with_cross_entropy(self.forward_without_softmax(X), y)
        self.conv1.backward(self.relu1.backward(self.maxpool1.backward(self.conv2.backward(self.relu2.backward(self.maxpool2.backward(self.flattener.backward(self.fully_connected.backward(d_out))))))))
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        return np.argmax(softmax(self.forward_without_softmax(X)), axis=1)

    def params(self):
        result = {
            'conv1_W' : self.conv1.params()['W'],
            'conv1_B': self.conv1.params()['B'],
            'conv2_W': self.conv2.params()['W'],
            'conv2_B': self.conv2.params()['B'],
            'fully_connected_W': self.fully_connected.params()['W'],
            'fully_connected_B': self.fully_connected.params()['B'],
        }
        return result
