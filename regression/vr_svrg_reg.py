import numpy as np
import math
from random import choice
from loss_function import squared_loss

from sklearn.base import BaseEstimator, RegressorMixin


class svrg_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, dim, round=1, step_size=0.1, temp=1.0):
        self.round = round
        self.step_size = step_size
        self.samples = []
        self.dim = dim
        self.temp = temp

    def fit(self, X_train, y_train):
        d = self.dim
        b = 10
        n = len(y_train)
        T = n * self.round
        h = self.step_size
        D = self.temp
        K = n / b

        samples = self.samples
        theta = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        samples.append(theta)
        # print('svrgfit, step-size=:' + str(h) + ' tmp=' + str(D)+ 'samples'+str(len(samples)))

        moments = []
        p = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        moments.append(p)

        g = np.zeros(d)
        w = np.zeros(d)

        # print('Total number of iters: ', T)
        for t in range(T):
            # if t % 1000 is 0:
            #     print('Iter ', t)

            theta = samples[t]
            if t % K == 0:
                tmp = np.zeros(d)
                for i in range(n):
                    x = X_train[i, :]
                    y = y_train[i]
                    tmp = tmp + (np.dot(theta, x) - y) * x
                g = theta + tmp
                w = theta

            I = []
            for i in range(b):
                I.append(choice(range(n)))

            tmp = np.zeros(d)
            for i in I:
                tmp = tmp + (np.dot(theta, X_train[i, :]) - y_train[i]) * X_train[i, :] \
                      - (np.dot(w, X_train[i, :]) - y_train[i]) * X_train[i, :]
            nabla = theta + float(n) / float(b) * tmp + g

            p_next = (1 - D * h) * moments[t] - h * nabla + math.sqrt(2 * D * h) \
                                                            * np.random.multivariate_normal(np.zeros(d), np.identity(d))
            theta_next = samples[t] + h * p_next
            samples.append(theta_next)
            moments.append(p_next)
        # print('score='+ str(self.score(X_train, y_train)))
        return self

    def score(self, X, y):
        sum = 0.
        n = len(y)
        # dn = 1.0 / n
        for i in range(n):
            sum += squared_loss(self.predict(X[i, :]), y[i])
        return -sum / n

    def predict(self, x):
        n = len(self.samples)
        # dn = 1.0 / n
        if n is 0:
            return 0.

        pred = 0.
        for theta in self.samples:
            pred += np.dot(x, theta)
        pred = pred / n
        return pred

    def fit2plot(self, X_train, X_test, y_train, y_test, ini_theta):
        self.samples = []
        mse = []
        lenTest = len(y_test)
        emp_pred_val = np.zeros(lenTest)
        realmse = 0
        empsum = 0

        d = self.dim
        b = 10
        n = len(y_train)
        T = n * self.round
        h = self.step_size
        D = self.temp
        K = n / b
        # print('svrg, step-size=:'+str(h)+' tmp='+str(D))

        samples = self.samples
        theta = ini_theta
        samples.append(theta)

        moments = []
        p = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        moments.append(p)

        g = np.zeros(d)
        w = np.zeros(d)

        # print('Plot total number of iters: ', T)
        for t in range(T):
            # if t % 1000 is 0:
            #     print('Plot iter: ', t)

            theta = samples[t]
            if t % K == 0:
                tmp = np.zeros(d)
                for i in range(n):
                    x = X_train[i, :]
                    y = y_train[i]
                    tmp = tmp + (np.dot(theta, x) - y) * x
                g = theta + tmp
                w = theta

            I = []
            for i in range(b):
                I.append(choice(range(n)))

            tmp = np.zeros(d)
            for i in I:
                tmp = tmp + (np.dot(theta, X_train[i, :]) - y_train[i]) * X_train[i, :] \
                      - (np.dot(w, X_train[i, :]) - y_train[i]) * X_train[i, :]
            nabla = theta + float(n) / float(b) * tmp + g

            p_next = (1 - D * h) * moments[t] - h * nabla + math.sqrt(2 * D * h) \
                                                            * np.random.multivariate_normal(np.zeros(d), np.identity(d))
            theta_next = samples[t] + h * p_next

            gap = 10
            if t % gap is 0:
                thetahere = samples[-gap:]
                lengap = len(thetahere)
                for i in range(lenTest):
                    for j in range(lengap):
                        emp_pred_val[i] += np.dot(X_test[i, :], thetahere[j])
                realmse += 1
                empsum += lengap
                emp_avg_val = emp_pred_val / empsum
                err = 0.0
                for i in range(lenTest):
                    err += squared_loss(emp_avg_val[i], y_test[i])
                err /= lenTest
                mse.append(err)

            samples.append(theta_next)
            moments.append(p_next)

        return mse


'''           
            gap = 10
            if t % gap is 0:
                thetahere = samples[-gap:]

                lengap = len(thetahere)
                emp_pred_val = np.zeros(lenTest)
                ileng = 1.0 / lengap
                for i in range(lenTest):
                    for j in range(lengap):
                        emp_pred_val[i] += (np.dot(X_test[i, :], thetahere[j]) * ileng)
                # emp_pred_val /= gap
                realmse += 1
                empsum += lengap
                emp_avg_val = (lengap / empsum) * emp_pred_val \
                              + ((empsum - lengap) / empsum) * emp_avg_val
                err = 0.0
                ilent = 1.0 / lenTest
                for i in range(lenTest):
                    err += (squared_loss(emp_avg_val[i], y_test[i]) * ilent)
                # err /= lenTest
                mse.append(err)
                print(t, err)
'''
