import numpy as np
from xlrd import open_workbook


def data_params():
    wb = open_workbook('data/1concrete/Concrete_Data.xls')
    data = []
    ans = []
    for s in wb.sheets():
        for row in range(1, s.nrows):
            col_value = []
            for col in range(s.ncols - 1):
                value = s.cell(row, col).value
                col_value.append(float(value))
            data.append(col_value)
            ans.append(s.cell(row, s.ncols - 1).value)

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
    saga2nd_params = {'step_size': [0.001, 0.005], \
                   'temp': [1, 3, 5]}

    return 'data/1concrete', len(X[0]), X_normed, y, saga_params, svrg_params, \
           sgld_params, sald_params, svrg2nd_params, saga2nd_params
