import numpy
import math
from random import choice
from loss_function import squared_loss

from sklearn.base import BaseEstimator, RegressorMixin

class saga_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, dim, round = 1, step_size = 0.1, temp = 1.0):
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

        samples = self.samples
        theta = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
        samples.append(theta)

        moments = []
        p = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
        moments.append(p)

        alpha = []
        for i in range(n):
            alpha.append(theta)

        g = numpy.zeros(d)
        for i in range(n):
            g = g - (y_train[i] - numpy.dot(alpha[i], X_train[i, :])) * X_train[i, :]

        print('Total number of iters: ', T)
        for t in range(T):
            if t % 1000 is 0:
                print('Iter: ', t)

            theta = samples[t]
            p = moments[t]

            I = []
            for i in range(b):
                I.append(choice(range(n)))
            tmp = numpy.zeros(d)
            for i in I:
                tmp = tmp + (numpy.dot(theta, X_train[i, :]) - y_train[i]) * X_train[i, :] \
                        - (numpy.dot(alpha[i], X_train[i, :]) - y_train[i]) * X_train[i, :]
            nabla = theta + float(n) / float(b) * tmp + g

            p_next = (1 - D*h) * p - h * nabla + math.sqrt(2*D*h) \
                        * numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
            theta_next = theta + h * p_next
            samples.append(theta_next)
            moments.append(p_next)

            for i in I:
                alpha[i] = theta
            g = g + tmp

        return self

    def score(self, X, y):
        sum = 0.
        n = len(y)
        for i in range(n):
            sum = sum + squared_loss(self.predict(X[i, :]), y[i])
        return -sum / n

    def predict(self, x):
        n = len(self.samples)
        if n is 0:
            return 0.

        pred = 0.
        for theta in self.samples:
            pred = pred + numpy.dot(x, theta)
        pred = pred / n
        return pred

    def fit2plot(self, X_train, X_test, y_train, y_test):
        self.samples = []
        mse = []

        d = self.dim
        b = 10
        n = len(y_train)
        T = n * self.round
        h = self.step_size
        D = self.temp

        samples = self.samples
        theta = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
        samples.append(theta)

        moments = []
        p = numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
        moments.append(p)

        alpha = []
        for i in range(n):
            alpha.append(theta)

        g = numpy.zeros(d)
        for i in range(n):
            g = g - (y_train[i] - numpy.dot(alpha[i], X_train[i, :])) * X_train[i, :]

        print('Plot total number of iters: ', T)
        for t in range(T):
            if t % 100 is 0:
                print('Plot iter: ', t)

            theta = samples[t]
            p = moments[t]

            I = []
            for i in range(b):
                I.append(choice(range(n)))
            tmp = numpy.zeros(d)
            for i in I:
                tmp = tmp + (numpy.dot(theta, X_train[i, :]) - y_train[i]) * X_train[i, :] \
                        - (numpy.dot(alpha[i], X_train[i, :]) - y_train[i]) * X_train[i, :]
            nabla = theta + float(n) / float(b) * tmp + g

            p_next = (1 - D*h) * p - h * nabla + math.sqrt(2*D*h) \
                        * numpy.random.multivariate_normal(numpy.zeros(d), numpy.identity(d))
            theta_next = theta + h * p_next
            samples.append(theta_next)
            moments.append(p_next)

            for i in I:
                alpha[i] = theta
            g = g + tmp

            if t % 10 is 0:
                err = - self.score(X_test, y_test)
                mse.append(err)

        return mse
