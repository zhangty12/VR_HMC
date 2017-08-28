from run_expr import run_expr

import init_1pima
import init_2diabetic
import init_3eeg

run_expr(init_3eeg, 1, 3, 1)  # initiate, rnd=3, cv=3, size=0
