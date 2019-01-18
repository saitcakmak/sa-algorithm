import numpy as np
import matplotlib.pyplot as plt
from mm1_toy import mm1


x = np.arange(5, 15, 0.1)
theta = 10
res = []
a = 0.005
M = 2000
for i in x:
    np.random.seed()
    out_l = []
    print("i: ", i)
    for j in range(1000):
        out, der = mm1(theta, i, a=a, M=M)
        out_l.append(out)
    res.append(np.average(out_l))


# rev = x * theta[0] * np.exp(-x/theta[1])

plt.plot(x, res)
plt.title("a = " + str(a) + " M = " + str(M))
# plt.plot(x, - rev)
# plt.plot(x, res - rev)
plt.show()
