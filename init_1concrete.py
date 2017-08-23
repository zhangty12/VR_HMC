import numpy
from xlrd import open_workbook

from init import initiator

class init_1concrete(initiator):
    def __init__(self):
        pass

    def data_params(self):
        wb = open_workbook('data/1concrete/Concrete_Data.xls')
        data = []
        ans = []
        for s in wb.sheets():
            for row in range(1, s.nrows):
                col_value = []
                for col in range(s.ncols-1):
                    value  = s.cell(row, col).value
                    col_value.append(float(value))
                data.append(col_value)
                ans.append(s.cell(row, s.ncols-1).value)

        X = numpy.array(data)
        y = numpy.array(ans)
        X_normed = X / X.max(axis = 0)

        saga_params = {'step_size': [0.005, 0.01, 0.05], \
                        'temp': [3, 5, 7]}
        svrg_params = {'step_size': [0.005, 0.01, 0.05], \
                        'temp': [3, 5, 7]}
        sgld_params = {'step_size': [0.00005, 0.0001, 0.0005]}
        sald_params = {'step_size': [0.00005, 0.0001, 0.0005]}

        return 'data/1concrete', len(data[0]), X_normed, y, saga_params, \
                svrg_params, sgld_params, sald_params
