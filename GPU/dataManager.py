from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import MinMaxScaler

def sharedArray(features, targets=None, validation_split=None):
    # input data must be [N samples, N features] matrix
    if targets is not None:
        if validation_split:
            # place data into shared variables
            N_val = int(np.ceil(validation_split*len(targets)))
            N_train = len(targets) - N_val

            T_x = theano.shared(
                np.asarray(features[:N_train,:], dtype=theano.config.floatX), borrow=True)
            T_y = theano.shared(
                np.asarray(np.vstack(targets[:N_train]), dtype=theano.config.floatX), borrow=True)

            # place data into shared variables
            V_x = theano.shared(
                np.asarray(features[N_train:,:], dtype=theano.config.floatX), borrow=True)
            V_y = theano.shared(
                np.asarray(np.vstack(targets[N_train:]), dtype=theano.config.floatX), borrow=True)

            return [T_x, T_y], [V_x, V_y], N_train, N_val
        else:
            # place data into shared variables
            shared_x = theano.shared(
                np.asarray(features, dtype=theano.config.floatX), borrow=True)
            shared_y = theano.shared(
                np.asarray(np.vstack(targets), dtype=theano.config.floatX), borrow=True)

            return shared_x, shared_y
    else:
        shared_x = theano.shared(
            np.asarray(features, dtype=theano.config.floatX), borrow=True)
        return shared_x

def standardize(X, Xtrain, axis=0):
    if axis==1:
        X = X.T
        Xtrain = Xtrain.T

    Xavg = np.mean(Xtrain,0)
    Xstd = np.std(Xtrain,0)

    return (X-Xavg) / Xstd

def center(X, Xtrain, axis=0):
    if axis==1:
        X = X.T
        Xtrain = Xtrain.T

    Xavg = np.mean(Xtrain,0)
    Xstd = np.std(Xtrain,0)

    return (X-Xavg)

def scale(X, Xtrain, axis=0):
    if axis==1:
        X = X.T
        Xtrain = Xtrain.T

    scaler = MinMaxScaler().fit(Xtrain)

    return scaler.transform(X)
