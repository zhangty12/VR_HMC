import math


def sigmoid(z):
    tmp = min(0, z)
    res = math.exp(tmp) / (math.exp(tmp) + math.exp(tmp - z))
    return res
