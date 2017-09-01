import numpy as np
import pandas as pd
import csv


def data_params():
    df = pd.read_csv(open('data/5protein/Protein_CASP.csv', encoding='utf-8'))
    dfArr = np.array(df.as_matrix())

    X = dfArr[:, 1:]
    y = dfArr[:, 0]
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

    return 'data/5protein', len(X[0]), X_normed, y, saga_params, svrg_params, \
           sgld_params, sald_params, svrg2nd_params, saga2nd_params
