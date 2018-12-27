import numpy as np
from old_notation_code.investment_params import b, c1, c2, base

theta = [0.7522, 0.5212, 1.3921, 0.3893, 1.0155]

n = 1000000
mu_gamma = 0.1
std_gamma = 0.05
cap = sum(theta)
val_base = c1 * cap + c2 * cap * cap
val_list = []

for i in range(n):
    print("iter: ", i)
    gamma = np.random.randn() * std_gamma + mu_gamma
    r = base + gamma * b
    val = val_base - np.dot(theta, r)
    val_list.append(val)

sorted_val = np.sort(val_list)
cvar = np.average(val_list[int(0.9 * n):])
print("cvar: ", cvar)
