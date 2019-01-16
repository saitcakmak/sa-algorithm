import numpy as np
import matplotlib.pyplot as plt
import two_sided_queue


x = np.arange(10, 25, 0.25)
theta = [40, 0.1, 10, 0.05]
res = []
for i in x:
    np.random.seed()
    out_l = []
    print("i: ", i)
    for j in range(4000):
        out, der = two_sided_queue.two_sided_ext(theta, i)
        out_l.append(out)
    res.append(np.average(out_l))


# rev = x * theta[0] * np.exp(-x/theta[1])

plt.plot(x, res)
# plt.plot(x, - rev)
# plt.plot(x, res - rev)
plt.show()
