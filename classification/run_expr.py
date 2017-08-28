from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

from vr_saga_reg import saga_estimator
from vr_svrg_reg import svrg_estimator
from vr_sgld_reg import sgld_estimator
from vr_sald_reg import sald_estimator


def run_expr(initiate, rnd=3, cv=3, size=0):
    name, dim, X, y, saga_params, svrg_params, sgld_params, sald_params = initiate.data_params()
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=1.0 / 5, random_state=10)

    if size != 0:
        X_train, X_va, y_train, y_va = train_test_split(X_tv, y_tv, test_size=1.0 / 8, random_state=10)

    # rnd = 3
    # cv = 3
    lenTrain = len(y_tv)

    saga = saga_estimator(dim=dim, round=rnd)
    svrg = svrg_estimator(dim=dim, round=rnd)
    sgld = sgld_estimator(dim=dim, round=rnd)
    sald = sald_estimator(dim=dim, round=rnd)

    cv_saga = GridSearchCV(estimator=saga, param_grid=saga_params, cv=cv)
    cv_svrg = GridSearchCV(estimator=svrg, param_grid=svrg_params, cv=cv)
    cv_sgld = GridSearchCV(estimator=sgld, param_grid=sgld_params, cv=cv)
    cv_sald = GridSearchCV(estimator=sald, param_grid=sald_params, cv=cv)

    if size != 0:
        cv_saga.fit(X_va, y_va)
        cv_svrg.fit(X_va, y_va)
        cv_sgld.fit(X_va, y_va)
        cv_sald.fit(X_va, y_va)
    else:
        cv_saga.fit(X_tv, y_tv)
        cv_svrg.fit(X_tv, y_tv)
        cv_sgld.fit(X_tv, y_tv)
        cv_sald.fit(X_tv, y_tv)

    print('saga params: ', cv_saga.best_params_)
    print('svrg params: ', cv_svrg.best_params_)
    print('sgld params: ', cv_sgld.best_params_)
    print('sald params: ', cv_sald.best_params_)

    ini_theta = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    saga_plot = cv_saga.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    svrg_plot = cv_svrg.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    sgld_plot = cv_sgld.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    sald_plot = cv_sald.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)

    plt.ylabel('Test MSE')
    plt.xlabel('Number of pass through data')
    lenT = len(svrg_plot)

    times = [10.0 / lenTrain * i for i in range(lenT)]
    # times = [i / lenTrain  for i in range(lenT)]

    plt.plot(times, saga_plot, 'r-', label='SAGA')
    plt.plot(times, svrg_plot, 'b-', label='SVRG')
    plt.plot(times, sgld_plot, 'g-', label='SGLD')
    plt.plot(times, sald_plot, 'y-', label='SALD')

    plt.legend()
    plt.savefig(name + '/loglh_' + name.split('/')[1] + '.png')
    plt.show()
