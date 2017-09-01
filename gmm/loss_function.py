import math


def lh(x, theta):
    sigma = 5
    gamma = 20
    prob = 1.0 / 2
    isigma = 1.0 / (2 * (sigma ** 2))
    t1 = ((x - theta) ** 2) * isigma
    t2 = ((x + theta - gamma) ** 2) * isigma
    tmp = min(0, t1, t2)
    res = (prob / (sigma * math.sqrt(2 * math.pi))) * \
          ((math.exp(tmp - t1) + math.exp(tmp - t2)) / math.exp(tmp))
    return res
