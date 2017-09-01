import numpy as np
import math


def data_params():
    np.random.seed(10)
    n = 20000
    a = np.array(np.random.rand(n))
    prob = 1.0 / 2
    mu1, mu2, sigma = -5, 25, 5
    dim = 1
    X = np.zeros(n)
    for i in range(n):
        if a[i] < prob:
            X[i] = np.random.normal(mu1, sigma)
        else:
            X[i] = np.random.normal(mu2, sigma)

    # X_normed = X / X.max(axis=0)
    # y_normed = y / y.max(axis=0)

    saga_params = {'step_size': [0.01], \
                   'temp': [3]}
    svrg_params = {'step_size': [0.01], \
                   'temp': [500]}
    sgld_params = {'step_size': [0.0005]}
    sald_params = {'step_size': [0.0005]}

    return 'gmm', dim, X, \
           saga_params, svrg_params, sgld_params, sald_params
