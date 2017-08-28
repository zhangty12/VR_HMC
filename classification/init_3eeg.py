import numpy as np
import re


def data_params():
    flag = 1
    data = []
    ans = []
    with open('data/3eeg/EEG_Eye_State.arff', encoding='utf-8') as f:
        for line in f:
            if flag == 1:
                line = list(filter(None, re.split('@|\n', line)))
                if len(line) > 0 and line[0] == 'DATA':
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
    sgld_params = {'step_size': [0.0005]}
    sald_params = {'step_size': [0.0005]}

    return 'data/3eeg', len(X[0]), X_normed, y, \
           saga_params, svrg_params, sgld_params, sald_params
