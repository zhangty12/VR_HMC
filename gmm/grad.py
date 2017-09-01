import math


def gradlog(x, y, theta):
    sigma = 5
    gamma = 20
    t1 = (x - theta) ** 2
    t2 = (x + theta - gamma) ** 2
    tmp = min(0.0, t1, t2)
    res = ((x - theta) * math.exp(tmp - t1) - (x + theta - gamma) * math.exp(tmp - t2)) / \
          ((sigma ** 2) * (math.exp(tmp - t1) + math.exp(tmp - t2)))
    return -res
