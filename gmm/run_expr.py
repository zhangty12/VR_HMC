from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import math

from vr_saga_reg import saga_estimator
from vr_svrg_reg import svrg_estimator
from vr_sgld_reg import sgld_estimator
from vr_sald_reg import sald_estimator


def run_expr(initiate, rnd=3, cv=3, size=0):
    name, dim, X, saga_params, svrg_params, sgld_params, sald_params = initiate.data_params()
    X_tv, X_test, y_tv, y_test = train_test_split(X, X, test_size=1.0 / 5, random_state=10)

    if size != 0:
        X_train, X_va, y_train, y_va = train_test_split(X_tv, y_tv, test_size=1.0 / 8, random_state=10)

    # rnd = 3
    # cv = 3

    # saga = saga_estimator(dim=dim, round=rnd)
    svrg = svrg_estimator(dim=dim, round=rnd)
    # sgld = sgld_estimator(dim=dim, round=rnd)
    # sald = sald_estimator(dim=dim, round=rnd)

    # cv_saga = GridSearchCV(estimator=saga, param_grid=saga_params, cv=cv)
    cv_svrg = GridSearchCV(estimator=svrg, param_grid=svrg_params, cv=cv)
    # cv_sgld = GridSearchCV(estimator=sgld, param_grid=sgld_params, cv=cv)
    # cv_sald = GridSearchCV(estimator=sald, param_grid=sald_params, cv=cv)

    if size != 0:
        # cv_saga.fit(X_va, y_va)
        cv_svrg.fit(X_va, y_va)
        # cv_sgld.fit(X_va, y_va)
        # cv_sald.fit(X_va, y_va)
    else:
        # cv_saga.fit(X_tv, y_tv)
        cv_svrg.fit(X_tv, y_tv)
        # cv_sgld.fit(X_tv, y_tv)
        # cv_sald.fit(X_tv, y_tv)

    # print('saga params: ', cv_saga.best_params_)
    print('svrg params: ', cv_svrg.best_params_)
    # print('sgld params: ', cv_sgld.best_params_)
    # print('sald params: ', cv_sald.best_params_)

    ini_theta = np.random.multivariate_normal(np.zeros(dim), np.identity(dim))
    # saga_plot = cv_saga.best_estimator_.fit2plot(X, X, ini_theta)
    svrg_plot = cv_svrg.best_estimator_.fit2plot(X, X, ini_theta)
    # sgld_plot = cv_sgld.best_estimator_.fit2plot(X, X, ini_theta)
    # sald_plot = cv_sald.best_estimator_.fit2plot(X, X, ini_theta)

    # plot

    sigma = 5
    gamma = 20
    prob = 1.0 / 2
    mu = np.arange(-50, 55, 0.01)
    y = np.zeros(len(mu))
    n = len(X)
    for i in range(n):
        y += np.log(prob / (sigma * np.sqrt(2 * np.pi)) * (
        np.exp(-(mu - X[i]) ** 2 / (2 * (sigma ** 2))) + np.exp(-(mu - (-X[i] + gamma)) ** 2 / (2 * (sigma ** 2)))))
    y += (np.log(1 / np.sqrt(2 * np.pi)) - (mu ** 2) / 2)
    plt.figure(1)
    plt.xlabel(r'$\mu$')
    plt.ylabel('Log-posterior')
    plt.title('Posterior')
    plt.plot(mu, y, 'r-')
    # plt.legend()
    plt.savefig(name + '_post' + '.png')

    # plt.figure(2)
    # plt.xlabel(r'$\mu$')
    # plt.ylabel('Sample count')
    # plt.title('Estimated Posterior')
    # plt.hist(saga_plot, bins=100)
    # plt.legend()
    # plt.savefig(name + '_post_' + 'SAGA' + '.png')

    plt.figure(3)
    plt.xlabel(r'$\mu$')
    plt.ylabel('Sample count')
    plt.title('Estimated Posterior')
    plt.hist(svrg_plot, bins=100)
    # plt.legend()
    plt.savefig(name + '_post_' + 'SVRG' + '.png')

    # plt.figure(4)
    # plt.xlabel(r'$\mu$')
    # plt.ylabel('Sample count')
    # plt.title('Estimated Posterior')
    # plt.hist(sgld_plot, bins=100)
    # plt.legend()
    # plt.savefig(name + '_post_' + 'SGLD' + '.png')
    #
    # plt.figure(5)
    # plt.xlabel(r'$\mu$')
    # plt.ylabel('Sample count')
    # plt.title('Estimated Posterior')
    # plt.hist(sald_plot, bins=100)
    # plt.legend()
    # plt.savefig(name + '_post_' + 'SALD' + '.png')

    # plt.figure(6)
    # plt.xlabel('x')
    # plt.ylabel('Sample count')
    # plt.title('Estimated Posterior')
    # plt.hist(svrg_ss, bins=100)
    # # plt.legend()
    # plt.savefig(name + '_post_' + 'ssSVRG' + '.png')

    plt.show()
