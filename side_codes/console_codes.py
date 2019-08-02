"""
this is just a file to copy-paste from console.
It is for plotting the comparison plot
"""

MLE = [0.0864636,0.049663]
mle_vals = value_plotter.plotter(MLE)[1]

a = np.sort(results, 0)
var_05 = a[49]
var_08 = a[79]
cvar_05 = np.average(a[49:], 0)
cvar_08 = np.average(a[79:], 0)
results = []
for i in range(100):
    results.append(pool_results[i][1])

results = np.array(results)
mean = np.average(results, 0)


for i in range(100):
    y = pool_results[i][1]
    plt.plot(x, y, color='green', linewidth=0.25, alpha=0.6)

plt.plot(x, correct_out[1], color='blue', label='Correct', lw=2)

plt.plot(x, mean, color='magenta', label='Expectation', lw=1.5)

plt.plot(x, mle_vals, color='blueviolet', label='MLE', lw=1.5)

plt.plot(x, var_05, color='red', label='VaR$_{0.5}$', lw=1.5)

plt.plot(x, var_08, color='darkmagenta', label='VaR$_{0.8}$', lw=1.5)

plt.plot(x, cvar_05, color='navy', label='CVaR$_{0.5}$', lw=1.5)

plt.plot(x, cvar_08, color='orangered', label='CVaR$_{0.8}$', lw=1.5)

plt.ylim((-20, 10))
plt.xlabel("Price")
plt.ylabel("Objective Value")
plt.title("Objective Comparison for Various Functions")
plt.legend(loc="lower left")
