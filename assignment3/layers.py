import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        predictions = predictions[np.newaxis, :]
    return np.exp(predictions - np.max(predictions, axis=1, keepdims=True)) / np.sum(
        np.exp(predictions - np.max(predictions, axis=1, keepdims=True)), axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if len(probs.shape) == 1:
        probs = probs[np.newaxis, :]
    target_index_arr = np.zeros_like(probs)
    target_index_arr[np.arange(len(probs)), target_index] = 1
    return -np.mean(np.log(probs[np.arange(len(probs)), target_index]))
    # return -np.sum(np.log(probs[np.arange(len(probs)), target_index]))


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    if len(predictions.shape) == 1:
        predictions = predictions[np.newaxis, :]
    target_index_arr = np.zeros_like(predictions)
    target_index_arr[np.arange(len(predictions)), target_index] = 1
    loss = cross_entropy_loss(softmax(predictions), target_index)
    dprediction = (softmax(predictions) - target_index_arr) / len(target_index_arr)
    # dprediction = (softmax(predictions) - target_index_arr)
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return (np.abs(X) + X) / 2

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return d_out * (np.abs(self.X) + self.X) / (2 * np.abs(self.X) + 1e-7)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        dx = d_out @ self.W.value.T
        dw = self.X.T @ d_out
        db = np.sum(d_out, axis=0)
        self.B.grad += db
        self.W.grad += dw
        return dx

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.X = None
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        X_padding = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
        batch_size, height, width, channels = X_padding.shape
        self.X = X_padding
        # out_height = 0
        # out_width = 0
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                result[:, y, x] = X_padding[:, y:y + self.filter_size, x:x + self.filter_size].reshape(
                    (batch_size, -1)) @ self.W.value.reshape((-1, self.out_channels))
        return result + self.B.value

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        dx = np.zeros_like(self.X)
        self.B.grad = np.sum(d_out.reshape((-1, out_channels)), axis=0)
        for y in range(out_height):
            for x in range(out_width):
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                dx[:, y:y + self.filter_size, x:x + self.filter_size, :] += (d_out[:, y, x] @ self.W.value.reshape((-1, out_channels)).T).reshape((batch_size, self.filter_size, self.filter_size, channels))
                self.W.grad += (self.X[:, y:y + self.filter_size, x:x + self.filter_size].reshape((batch_size, -1)).T @ d_out[:, y, x]).reshape((self.filter_size, self.filter_size, channels, out_channels))
        if self.padding > 0:
            return dx[:,self.padding:-self.padding,self.padding:-self.padding,:]
        return dx
    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = int((height - self.pool_size)/self.stride + 1)
        out_width = int((width - self.pool_size)/self.stride + 1)
        result = np.zeros((batch_size, out_height, out_width, channels))
        for y in range(out_height):
            for x in range(out_width):
                result[:, y, x] = np.max(np.max(X[:, y*self.pool_size:(y+1)*self.pool_size, x*self.pool_size:(x+1)*self.pool_size], axis=1), axis=1)
        return result


    def backward(self, d_out):
        _, out_height, out_width, out_channels = d_out.shape
        dx = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                dx[:, y*self.pool_size:(y+1)*self.pool_size, x*self.pool_size:(x+1)*self.pool_size] += d_out[:,y,x][:,np.newaxis, np.newaxis] * (self.X[:, y*self.pool_size:(y+1)*self.pool_size, x*self.pool_size:(x+1)*self.pool_size] == np.max(np.max(self.X[:, y*self.pool_size:(y+1)*self.pool_size, x*self.pool_size:(x+1)*self.pool_size], axis=1), axis=1)[:,np.newaxis, np.newaxis])
        return dx

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, -1))

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
