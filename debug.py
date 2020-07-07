from normal_runner import analytic_value_VaR
print(analytic_value_VaR(0.49748))
from normal_runner import estimate
print(estimate(0.49748, 100000, 0.5, 'CVaR', -15, 10, 4, 2))

import numpy as np
import matplotlib.pyplot as plt


x_l = np.arange(-1, 1, 0.01)
res = np.empty_like(x_l)
for i in range(len(x_l)):
    out = estimate(x_l[i], 10000, 0.5, 'CVaR', -15, 10, 4, 2)
    res[i] = out[0]

plt.plot(x_l, res)
plt.show()
