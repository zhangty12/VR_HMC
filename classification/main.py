from run_expr import run_expr
import matplotlib.pyplot as plt

import init_1pima
import init_2diabetic
import init_3eeg


plt.figure(1)
run_expr(init_1pima, rnd=2, cv=2, size=1)  # initiate, rnd=3, cv=3, size=0

plt.figure(2)
run_expr(init_2diabetic, rnd=2, cv=2, size=1)

plt.figure(3)
run_expr(init_3eeg, rnd=1, cv=2, size=1)

