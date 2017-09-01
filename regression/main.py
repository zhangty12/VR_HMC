from run_expr import run_expr
import matplotlib.pyplot as plt

import init_1concrete
import init_2noise
import init_3parkinson
import init_4bike
import init_5protein

print('1concrete')
plt.figure(1)
run_expr(init_1concrete, rnd=3, cv=3, size=1)  # initiate, rnd=3, cv=3, size=0

print('2noise')
plt.figure(2)
run_expr(init_2noise, rnd=5, cv=3, size=1)

print('3parkinson')
plt.figure(3)
run_expr(init_3parkinson, rnd=10, cv=3, size=1)

print('4bike')
plt.figure(4)
run_expr(init_4bike, rnd=3, cv=3, size=1)

print('5protein')
plt.figure(5)
run_expr(init_5protein, rnd=3, cv=3, size=1)

plt.show()
