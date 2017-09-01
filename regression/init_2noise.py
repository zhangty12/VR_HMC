import numpy as np


def data_params():
    data = []
    ans = []
    for line in open('data/2noise/noise.dat', 'r').readlines():
        datum_str = line.split()
        datum = [float(f) for f in datum_str]
        data.append(datum[:5])
        ans.append(datum[5])

    X = np.array(data)
    y = np.array(ans)
    X_normed = X / X.max(axis=0)
    # y_normed = y / y.max(axis=0)

    saga_params = {'step_size': [0.01, 0.001, 0.005], \
                   'temp': [1, 3, 5]}
    svrg_params = {'step_size': [0.01, 0.001, 0.005], \
                   'temp': [1, 3, 5]}
    sgld_params = {'step_size': [0.0005, 0.0001]}
    sald_params = {'step_size': [0.0005, 0.0001]}
    svrg2nd_params = {'step_size': [0.01, 0.001, 0.005], \
                      'temp': [1, 3, 5]}
    saga2nd_params = {'step_size': [0.01, 0.001, 0.005], \
                      'temp': [1, 3, 5]}

    return 'data/2noise', len(X[0]), X_normed, y, saga_params, svrg_params, \
           sgld_params, sald_params, svrg2nd_params, saga2nd_params
