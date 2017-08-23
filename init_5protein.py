import numpy as np
import pandas as pd
import csv

from init import initiator

class init_5protein(initiator):
    def __init__(self):
        pass

    def data_params(self):

        df = pd.read_csv(open('data/5protein/Protein_CASP.csv', encoding='utf-8'))
        dfArr = np.array(df.as_matrix())

        X = dfArr[:, 1:]
        y = dfArr[:, 0]
        X_normed = X / X.max(axis=0)

        saga_params = {'step_size': [0.005, 0.01, 0.05], \
                        'temp': [3, 5, 7]}
        svrg_params = {'step_size': [0.005, 0.01, 0.05], \
                        'temp': [3, 5, 7]}
        sgld_params = {'step_size': [0.00005, 0.0001, 0.0005]}
        sald_params = {'step_size': [0.00005, 0.0001, 0.0005]}

        return 'data/5protein', len(X[0]), X_normed, y, \
               saga_params, svrg_params, sgld_params, sald_params
