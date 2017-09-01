import numpy as np
import re


def data_params():
    flag = 1
    data = []
    ans = []
    with open('data/2diabetic/messidor_features.arff', encoding='utf-8') as f:
        for line in f:
            if flag == 1:
                line = list(filter(None, re.split('@|\n', line)))
                if len(line) > 0 and line[0] == 'data':
                    flag = 0
            else:
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
    sgld_params = {'step_size': [0.0001]}
    sald_params = {'step_size': [0.0001]}
    svrg2nd_params = {'step_size': [0.02], \
                      'temp': [3]}
    saga2nd_params = {'step_size': [0.02], \
                      'temp': [3]}

    return 'data/2diabetic', len(X[0]), X_normed, y, saga_params, svrg_params, \
           sgld_params, sald_params, svrg2nd_params, saga2nd_params
