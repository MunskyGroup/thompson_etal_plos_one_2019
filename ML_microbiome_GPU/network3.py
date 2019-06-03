"""network3.py
~~~~~~~~~~~~~~

MIT License

Copyright (c) 2012-2018 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import numpy.matlib
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool

# import dataManager for creating shared arrays
import dataManager
from dataManager import *


# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)

from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# Constants
GPU = False
if GPU:
    '''
    print("Trying to run under a GPU.  If this is not desired, then modify " + \
          "network3.py\nto set the GPU flag to False.")
    '''
    try:
        theano.config.device = 'gpu0'
    except:
        pass  # it's already set
    theano.config.floatX = 'float64'
else:
    '''
    print("Running with a CPU.  If this is not desired, then the modify " + \
          "network3.py to set\nthe GPU flag to True.")
    '''

np.random.seed(123)

# Main class used to construct and train networks
class Network(object):

    def __init__(self, NF=None, hidden_layers=1, nodes=15, activation_fn='sigmoid',
        eta=.03, decay=0.0, lmbda=0.0, patience=15, mini_batch_size=10, verbose=False,
        p_dropout=0.5):
        """Takes model architecture information and hyper parameters"""
        self.verbose = verbose
        self.NF = NF
        self.activation_fn=activation_fn
        self.nodes=nodes
        self.hidden_layers=hidden_layers
        self.lmbda = lmbda
        self.eta = eta
        self.decay = decay
        self.patience = patience
        self.p_dropout = p_dropout
        self.mini_batch_size = mini_batch_size

        self.layers = [
            FullyConnectedLayer(n_in=NF, n_out=self.nodes, activation_fn=activation_fn, p_dropout=p_dropout),
            FullyConnectedLayer(n_in=self.nodes, n_out=1, activation_fn='linear')]

        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def fit(self, X_train, Y_train, FS=False):
        training_data, validation_data, N_train, N_val = sharedArray(X_train, Y_train, validation_split=0.1)

        if FS:
            self.SGD(training_data, validation_data)
            return self.getFeatureImportance(training_data)
        else:
            self.SGD(training_data, validation_data)

    def SGD(self, training_data, validation_data):
        """Train the network using mini-batch stochastic gradient descent."""
        epochs = 500
        ep = T.lscalar()
        training_x, training_y = training_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data) // self.mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])

        cost = self.layers[-1].cost(self) + \
               0.5 * self.lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param - self.eta*np.exp(-self.decay*ep) * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch and compute validation accuracy
        i = T.lscalar()  # mini-batch index

        train_mb = theano.function(
            [i, ep], updates=updates,
            givens={
                self.x:
                    training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # Do the actual training
        best_val_loss = np.inf
        patience_count = 0

        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                train_mb(minibatch_index, epoch)

            val_loss = self.validation_loss(validation_data)
            if self.verbose:
                print("Epoch {0}: Validation Loss: {1}".format(
                    epoch, val_loss))

            if patience_count > self.patience:
                return

            if val_loss >= best_val_loss:
                patience_count+=1
            else:
                best_val_loss = val_loss

    def getFeatureImportance(self, training_data):

        training_x, training_y = training_data

        # calculate number of training batches
        num_training_batches = size(training_data) // self.mini_batch_size

        # define functions to train a mini-batch and compute validation accuracy
        grad_FS = T.grad(self.layers[-1].get_output(self), self.params[0])

        # define functions to train a mini-batch and compute validation accuracy
        i = T.lscalar()  # mini-batch index

        # define function which returns gradient values
        feature_importance = theano.function([i], grad_FS,
            givens={
                self.x:
                    training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # initialize feature importance vector as empty array with NF elements
        feature_importances = np.zeros([num_training_batches, self.NF])

        for minibatch_index in range(num_training_batches):
            # calculate gradient of model output wrt feature weights
            gradient_matrix = feature_importance(minibatch_index)
            feature_importances[minibatch_index, :] = np.mean(gradient_matrix, 1)

        return np.mean(feature_importances, 0)

    def validation_loss(self, validation_data):
        # function to compute MSE with validation data
        # this function works with data sets < mini_batch_size
        val_x, val_y = validation_data
        num_val_samples = val_x.get_value(borrow=True).shape[0]
        i = T.lscalar()

        loss = theano.function([], (self.layers[-1].y_out-val_y)**2,
            givens={self.x: val_x})

        return 0.5*np.mean(loss())

    def predict(self, X_test):
        test_x = sharedArray(X_test)
        num_test_batches = test_x.get_value(borrow=True).shape[0]
        i = T.lscalar()

        predict = theano.function([i], self.layers[-1].y_out,
            givens={self.x: test_x[i:i+1]})

        predictions = [predict(j) for j in range(num_test_batches)]

        return np.ravel(predictions)

    def get_params(self, deep=True):
        return {"verbose": self.verbose,
                "NF": self.NF,
                "decay": self.decay,
                "hidden_layers": self.hidden_layers,
                "nodes": self.nodes,
                "activation_fn": self.activation_fn,
                "p_dropout": self.p_dropout,
                "eta": self.eta,
                "lmbda": self.lmbda,
                "patience": self.patience,
                "mini_batch_size": self.mini_batch_size}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


#  Define layer types

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn='sigmoid', p_dropout=0.5):
        self.n_in = n_in
        self.n_out = n_out
        if activation_fn == 'sigmoid':
            self.activation_fn = sigmoid
        if activation_fn == 'linear':
            self.activation_fn = linear
        if activation_fn == 'tanh':
            self.activation_fn = tanh
        if activation_fn == 'relu':
            self.activation_fn = ReLU
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_in), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt #inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def get_output(self, net):
        "Return Feature Selection information"
        return T.mean(self.output_dropout)

    def cost(self, net):
        "Return the least-squares cost."
        return .5*T.mean((net.y-self.output_dropout)**2)


#### Miscellanea
def size(data):
    #"Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)
