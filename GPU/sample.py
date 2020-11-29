import numpy as np
import matplotlib.pyplot as plt


P1 = [  0,   4.,   2.,   1.,   1.,   1.,   1.,   1.,  12.,   2.,  10.,
         3.,   9.,   1.,   1.,   2.,   1.,   1.,   1.,   1.,   1.,   1.,
         4.,   1.,  16.,   1.,  22.,   2., 104.,   3.,  15.,  84.,   1.,
        12.,   1.,   3.,   2.,   1.,   9.,   2., 131., 151.,   4.,  12.,
         2., 108.,   1.,   3.,   1.,   2.,   1.,   8.,   2.,   6.,   3.,
        22.,   4.,   3.,   1.,   1.,  23.,  16.,   2.,   1.,   6.,   6.,
        2.,   1.,   1.,   4.,   1.,  20.,   1.,  22.,   2.,  11.,  19.,
        17.,   1.,   1.,   3.,   1.,   1.,   1.,   2.,   1.,   5.,  13.,
        2.,   2.,   1.,   2.,   1.,   1.,   1.,   1.,   1.,   1.,   2.,
        4.,   1.,   1.,   1.,   6.,   1.,   1.,   1.,   1.];


def randsample(k , w):
    # randomly sample k integers from 1 to n with acceptance probability
    # determined by weights w
    n = len(w)
    sample = np.zeros(k)
    i = 0
    while True:
        for j in range(n):
            u = np.random.uniform(0, 1)
            p = w[j] / np.max(w)
            if p > u:
                sample[i] = j
                i += 1
                if i >= k:
                    return sample
    return

def resample(x):
    # x represents the weighted probability of sampling an integer from 1:len(x)
    x = np.array(x)
    # initialize new sample = length(x) of zeros
    x2 = np.zeros(len(x))
    # randomly sample len(x) points where each point has p(i) = xi / max(x)
    sample_of_x = randsample(int(len(x)*np.mean(x)), x)
    jbin, wbin = np.unique(sample_of_x, return_counts = True)
    jbin = np.array(jbin, int)
    x2[jbin] = wbin
    return x2


def expand(X, n=9):
    # function calls sample, returns array with original sample and n resamplings
    # if x is matrix shape NSxNF, expand returns matrix shape NF*(n+1) x NF
    NS, NF = X.shape
    X_expand = np.zeros([NS*(n+1), NF])
    j = 0 # placement index for X_expand
    for sample in range(NS):
        x = X[sample, :]
        X_expand[j, :] = x
        j += 1
        for i in range(n):
            X_expand[j, :] = resample(x)
            j += 1
    return X_expand

P1 = np.array(P1).reshape(1, len(P1))
P2 = np.zeros([2, P1.shape[1]])
P2[0, :] = P1
P2[1, :] = P1
P = expand(P1)

#%%
plt.plot(P.T)
plt.show()
