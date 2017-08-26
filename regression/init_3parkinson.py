import numpy as np
import pandas as pd
import csv
import re


def data_params():
    flag = 1
    data = []
    ans = []
    with open('data/3parkinson/parkinsons_updrs.txt', encoding='utf-8') as f:
        for line in f:
            if flag == 1:
                flag = 0
            else:
                tmp = re.split(',', line)
                tmp = [float(s) for s in tmp]
                data.append(tmp[1:4] + tmp[6:])
                ans.append(tmp[4])  # 4 motor_UPDRS, 5 total_UPDRS

    X = np.array(data)
    y = np.array(ans)
    X_normed = X / X.max(axis=0)
    # y_normed = y / y.max(axis=0)

    saga_params = {'step_size': [0.01], \
                   'temp': [3]}
    svrg_params = {'step_size': [0.0001], \
                   'temp': [1]}
    sgld_params = {'step_size': [0.00005, 0.0001, 0.0005]}
    sald_params = {'step_size': [0.00005, 0.0001, 0.0005]}

    return 'data/3parkinson', len(X[0]), X_normed, y, \
           saga_params, svrg_params, sgld_params, sald_params
