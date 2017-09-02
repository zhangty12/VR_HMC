from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

from vr_saga_reg import saga_estimator
from vr_svrg_reg import svrg_estimator
from vr_sgld_reg import sgld_estimator
from vr_sald_reg import sald_estimator
from vr_svrg2nd_reg import svrg2nd_estimator
from vr_saga2nd_reg import saga2nd_estimator


def run_expr(initiate, rnd=3, cv=3, size=0):
    name, dim, X, y, saga_params, svrg_params, sgld_params, sald_params, \
    svrg2nd_params, saga2nd_params = initiate.data_params()
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
    svrg2nd = svrg2nd_estimator(dim=dim, round=rnd)
    saga2nd = saga2nd_estimator(dim=dim, round=rnd)

    cv_saga = GridSearchCV(estimator=saga, param_grid=saga_params, cv=cv)
    cv_svrg = GridSearchCV(estimator=svrg, param_grid=svrg_params, cv=cv)
    cv_sgld = GridSearchCV(estimator=sgld, param_grid=sgld_params, cv=cv)
    cv_sald = GridSearchCV(estimator=sald, param_grid=sald_params, cv=cv)
    cv_svrg2nd = GridSearchCV(estimator=svrg2nd, param_grid=svrg2nd_params, cv=cv)
    cv_saga2nd = GridSearchCV(estimator=saga2nd, param_grid=saga2nd_params, cv=cv)

    if size != 0:
        cv_saga.fit(X_va, y_va)
        cv_svrg.fit(X_va, y_va)
        cv_sgld.fit(X_va, y_va)
        cv_sald.fit(X_va, y_va)
        cv_svrg2nd.fit(X_va, y_va)
        cv_saga2nd.fit(X_va, y_va)
    else:
        cv_saga.fit(X_tv, y_tv)
        cv_svrg.fit(X_tv, y_tv)
        cv_sgld.fit(X_tv, y_tv)
        cv_sald.fit(X_tv, y_tv)
        cv_svrg2nd.fit(X_tv, y_tv)
        cv_saga2nd.fit(X_tv, y_tv)

    print('saga params: ', cv_saga.best_params_)
    print('svrg params: ', cv_svrg.best_params_)
    print('sgld params: ', cv_sgld.best_params_)
    print('sald params: ', cv_sald.best_params_)
    print('svrg2nd params: ', cv_svrg2nd.best_params_)
    print('saga2nd params: ', cv_saga2nd.best_params_)

    ini_theta = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    saga_plot = cv_saga.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    svrg_plot = cv_svrg.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    sgld_plot = cv_sgld.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    sald_plot = cv_sald.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    svrg2nd_plot = cv_svrg2nd.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)
    saga2nd_plot = cv_saga2nd.best_estimator_.fit2plot(X_tv, X_test, y_tv, y_test, ini_theta)

    plt.ylabel('Test MSE')
    plt.xlabel('Number of pass through data')
    lenT = len(svrg_plot)

    times = [10.0 / lenTrain * i for i in range(lenT)]
    # times = [i / lenTrain  for i in range(lenT)]

    ts = np.array(times)
    saga_plots = np.array(saga_plot)
    svrg_plots = np.array(svrg_plot)
    sgld_plots = np.array(sgld_plot)
    sald_plots = np.array(sald_plot)
    svrg2nd_plots = np.array(svrg2nd_plot)
    saga2nd_plots = np.array(saga2nd_plot)

    plots = np.row_stack((ts, saga_plots, svrg_plots,
                          sgld_plots, sald_plots, svrg2nd_plots, saga2nd_plots))
    plotdf = pd.DataFrame(plots)
    fileplot = name + '/plotdata_' + name.split('/')[1] + '.csv'
    plotdf.to_csv(fileplot, encoding='utf-8', index=False)

    plt.semilogy(times, saga_plot, 'r-', label='SAGA')
    plt.semilogy(times, svrg_plot, 'b-', label='SVRG')
    plt.semilogy(times, sgld_plot, 'g-', label='SGLD')
    plt.semilogy(times, sald_plot, 'y-', label='SALD')
    plt.semilogy(times, svrg2nd_plot, 'g-.', label='SVRG2nd')
    plt.semilogy(times, saga2nd_plot, 'y+', label='SAGA2nd')

    plt.legend()
    plt.savefig(name + '/mse_' + name.split('/')[1] + '.png')
    # plt.show()
