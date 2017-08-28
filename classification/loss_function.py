import math


def loglh(x, y):
    return (y * math.log(x) + (1 - y) * math.log(1 - x))
