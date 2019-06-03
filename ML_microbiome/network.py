"""network.py
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
"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np
from scipy.stats import multivariate_normal as mvn

np.random.seed(21)

#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


#### Main Network class
class Network(object):

    def __init__(self, NF, nodes=10, eta=.5, lmbda=.5, patience=10, verbose=False):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.verbose = verbose
        self.NF = NF
        # nodes should be passed as a list
        '''
        try:
            # this will err if nodes is not a list
            self.nodes = list(nodes)
        except:
            self.nodes = [nodes]
        self.sizes = sum([[NF], self.nodes, [1]], [])
        '''
        self.nodes = nodes
        self.eta = eta
        self.lmbda = lmbda
        self.patience = patience
        self.sizes = [NF, nodes, 1]
        self.num_layers = len(self.sizes)
        self.default_weight_initializer()
        self.cost=QuadraticCost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.
        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a)+b)
        # linear final layer
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        return a

    def fit(self, X, Y, tune=False):

        monitor_evaluation_cost=True,
        monitor_training_cost=True

        NS, NF = X.shape
        validation_split=0.15
        NS_val = int(validation_split*NS)
        training_inputs = [np.reshape(x, (NF, 1)) for x in X]
        training_results = [y for y in Y]
        training_data = list(zip(training_inputs, training_results))
        # partition some of the training data for validation
        random.shuffle(training_data)
        if NS_val > 0:
            evaluation_data = training_data[:NS_val]
        else:
            evaluation_data is None
        training_data = training_data[NS_val:]

        if tune:
            self.SGD(training_data, evaluation_data, monitor_training_cost=False)
        else:
            return self.SGD(training_data, evaluation_data)

    def SGD(self, training_data, evaluation_data, monitor_training_cost=True):

        epochs=200
        mini_batch_size=10
        eta = self.eta
        lmbda = self.lmbda
        patience = self.patience

        monitor_evaluation_cost = True
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost = []
        training_cost = []
        trial_num = []
        all_weights = []
        all_biases = []
        acc = 0
        trials = 0
        patience_count = 0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                trials += 1
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            if self.verbose: print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                trial_num.append(trials)
                if self.verbose: print("Cost on training data: {}".format(cost))
            if monitor_evaluation_cost:
                all_weights.append(self.weights)
                all_biases.append(self.biases)
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                if cost > np.min(evaluation_cost):
                    patience_count += 1
                if self.verbose: print("Cost on evaluation data: {}".format(cost))
                if patience_count >= patience:
                    self.weights = all_weights[np.argmin(evaluation_cost)]
                    self.biases = all_biases[np.argmin(evaluation_cost)]
                    return evaluation_cost[:np.argmin(evaluation_cost)+1], training_cost[:np.argmin(evaluation_cost)+1]

        self.weights = all_weights[np.argmin(evaluation_cost)]
        self.biases = all_biases[np.argmin(evaluation_cost)]
        return evaluation_cost[:np.argmin(evaluation_cost)+1], training_cost[:np.argmin(evaluation_cost)+1]

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # let the last activation be linear
        zs.append(np.dot(self.weights[-1], activation)+self.biases[-1])
        activations.append(zs[-1])
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            #sp = np.reshape(np.gradient(np.hstack(sigmoid(z)), np.hstack(z)), z.shape)
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def predict(self, data):
        NS, NF = data.shape
        yps = []
        for i, x in enumerate(data):
            x = np.reshape(x, (NF, 1))
            for b, w in zip(self.biases[:-1], self.weights[:-1]):
                x = sigmoid(np.dot(w, x)+b)
            yps.append(np.dot(self.weights[-1], x) + self.biases[-1])
        return np.ravel(yps)

    def feature_importance(self, X, Y):
        NS, NF = X.shape
        inputs = [np.reshape(x, (NF, 1)) for x in X]
        labels = [y for y in Y]
        data = list(zip(inputs, labels))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in data:
            delta_nabla_b, delta_nabla_w = self.backprop_output(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # account for sign changes that occur due to deeper layers
        downstream_affect = np.ones(nabla_w[0].shape[0])
        for w in self.weights[1:]:
            downstream_affect = downstream_affect * np.sign(w.T)

        weighted_importance = np.mean(nabla_w[0].T.dot(downstream_affect), 1)
        return weighted_importance / max(abs(weighted_importance))

    def backprop_output(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # let the last activation be linear
        zs.append(np.dot(self.weights[-1], activation)+self.biases[-1])
        activations.append(zs[-1])
        # backward pass
        delta = zs[-1]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            #sp = np.reshape(np.gradient(np.hstack(sigmoid(z)), np.hstack(z)), z.shape)
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def get_params(self, deep=True):
        return {"NF": self.NF,
                "nodes": self.nodes,
                "eta": self.eta,
                "lmbda": self.lmbda,
                "patience": self.patience,
                "verbose": self.verbose}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
