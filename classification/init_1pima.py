import numpy as np
import re


def data_params():
    data = []
    ans = []
    with open('data/1pima/pima-indians-diabetes.data', encoding='utf-8') as f:
        for line in f:
            tmp = re.split(',', line)
            tmp = [float(s) for s in tmp]
            data.append(tmp[:-1])
            ans.append(tmp[-1])

    X = np.array(data)
    y = np.array(ans)
    X_normed = X / X.max(axis=0)
    # y_normed = y / y.max(axis=0)

    saga_params = {'step_size': [0.01], \
                   'temp': [3]}
    svrg_params = {'step_size': [0.01], \
                   'temp': [3]}
    sgld_params = {'step_size': [0.0005]}
    sald_params = {'step_size': [0.0005]}

    return 'data/1pima', len(X[0]), X_normed, y, \
           saga_params, svrg_params, sgld_params, sald_params
