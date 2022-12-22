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
    return np.exp(predictions - np.max(predictions, axis=1, keepdims =True))/np.sum\
        (np.exp(predictions - np.max(predictions, axis=1, keepdims = True)),
         axis=1, keepdims = True)

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
    #return -np.mean(np.log(probs[np.arange(len(probs)), target_index]))
    return -np.sum(np.log(probs[np.arange(len(probs)), target_index]))


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
    #dprediction = (softmax(predictions) - target_index_arr)/len(target_index_arr)
    dprediction = (softmax(predictions) - target_index_arr)
    return loss, dprediction
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

    loss = reg_strength*np.sum(W**2)
    grad = 2*reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T @ dprediction
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            for batch_indices in batches_indices:
                loss_cross_entropy, dW_cross_entropy = linear_softmax(X[batch_indices], self.W, y[batch_indices])
                loss_regularization, dW_regularization = l2_regularization(self.W, reg)
                loss = loss_cross_entropy + loss_regularization
                self.W = self.W - learning_rate * (dW_cross_entropy + dW_regularization)
                loss_history.append(loss)
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        predictions = np.dot(X, self.W)
        return np.argmax(softmax(predictions), axis=1)



                
                                                          

            

                
