from __future__ import division
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
