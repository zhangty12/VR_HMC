import numpy as np
import math
from random import choice
from sklearn.base import BaseEstimator, RegressorMixin

from loss_function import lh
from grad import gradlog


class svrg_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, dim, round=1, step_size=0.1, temp=1.0):
        self.round = round
        self.step_size = step_size
        self.samples = []
        self.dim = dim

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
        print('svrgfit, step-size=:' + str(h) + ' tmp=' + str(D))

        moments = []
        p = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        moments.append(p)

        g = np.zeros(d)
        w = np.zeros(d)

        print('Total number of iters: ', T)
        for t in range(T):
            if t % 1000 is 0:
                print('Iter ', t)

            theta = samples[t]
            if t % K == 0:
                tmp = np.zeros(d)
                for i in range(n):
                    x = X_train[i]
                    y = y_train[i]
                    tmp += gradlog(x, y, theta)
                g = theta + tmp
                w = theta

            I = []
            for i in range(b):
                I.append(choice(range(n)))

            tmp = np.zeros(d)
            for i in I:
                tmp = tmp + gradlog(X_train[i], y_train[i], theta) \
                      - gradlog(X_train[i], y_train[i], w)
            nabla = theta + float(n) / float(b) * tmp + g

            p_next = (1 - D * h) * moments[t] - h * nabla + math.sqrt(2 * D * h) \
                                                            * np.random.multivariate_normal(np.zeros(d), np.identity(d))
            theta_next = samples[t] + h * p_next
            samples.append(theta_next)
            moments.append(p_next)
        print('score=' + str(self.score(X_train, y_train)))
        return self

    def score(self, X, y):
        sum = 0.
        n = len(y)
        # dn = 1.0 / n
        for i in range(n):
            sum += math.log(self.predict(X[i]))
        return sum / n
        # return np.random.rand()

    def predict(self, x):
        n = len(self.samples)
        # dn = 1.0 / n
        if n is 0:
            return 0.

        pred = 0.
        for theta in self.samples:
            pred += lh(x, theta)
        pred = pred / n
        return pred
        # return np.random.rand()

    def fit2plot(self, X_train, y_train, ini_theta):
        self.samples = []

        d = self.dim
        b = 10
        n = len(y_train)
        T = n * self.round
        h = self.step_size
        D = self.temp
        K = n / b

        print('svrgfit2plot, step-size=' + str(h) + ' tmp=' + str(D))

        samples = self.samples
        theta = ini_theta
        samples.append(theta)

        moments = []
        p = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        moments.append(p)

        g = np.zeros(d)
        w = np.zeros(d)

        print('Plot total number of iters: ', T)
        for t in range(T):
            if t % 1000 is 0:
                print('Plot iter: ', t)

            theta = samples[t]
            if t % K == 0:
                tmp = np.zeros(d)
                for i in range(n):
                    x = X_train[i]
                    y = y_train[i]
                    tmp += gradlog(x, y, theta)
                g = theta + tmp
                w = theta

            I = []
            for i in range(b):
                I.append(choice(range(n)))

            tmp = np.zeros(d)
            for i in I:
                tmp = tmp + gradlog(X_train[i], y_train[i], theta) \
                      - gradlog(X_train[i], y_train[i], w)
            nabla = theta + float(n) / float(b) * tmp + g

            p_next = (1 - D * h) * moments[t] - h * nabla + math.sqrt(2 * D * h) \
                                                            * np.random.multivariate_normal(np.zeros(d), np.identity(d))
            theta_next = samples[t] + h * p_next

            samples.append(theta_next)
            moments.append(p_next)

        # sigma = 5
        # gamma = 20
        # np.random.seed(10)
        # ns = len(samples)
        # prob = 1.0 / 2
        # ss = []
        # for i in range(ns):
        #     for j in range(10):
        #         a = np.random.rand()
        #         if a < prob:
        #             ss.append(np.random.normal(samples[i], sigma))
        #         else:
        #             ss.append(np.random.normal(-samples[i] + gamma, sigma))

        return np.array(samples)  #, np.array(ss)
