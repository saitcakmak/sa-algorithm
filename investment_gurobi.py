"""
this is an SAA implementation of investment problem based on v3
For reference in modeling, we use Qian2015Uncertainty-Framework pg 22-23
model (34) with CVaR replaced with expectation
"""
from gurobipy import *
import numpy as np
import datetime
from investment_params import b, c1, c2, base

start = datetime.datetime.now()
n, m = 1000, 4000
delta = 0.9
mu_gamma = 0.1
std_gamma = 0.06

# start gurobi model
model = Model("invest")
# actual decision variables
theta = model.addVars(range(5), name="theta", lb=-10000)
# the CVaR decision variable
alpha = model.addVar(name="alpha", lb=-10000)
# auxiliary decision variables for each CVaR sample
u = model.addVars(range(n), lb=0, name="u")
# set objective value
model.setObjective(alpha + (1/((1 - delta) * n)) * quicksum(u), GRB.MINIMIZE)

# constraints:
for i in range(n):
    now = datetime.datetime.now()
    print("iter: ", i, " time: ", (now-start))
    inner_sum = 0
    gamma = np.random.randn() * std_gamma + mu_gamma
    # generate inner expectation
    for j in range(m):
        r = np.random.multivariate_normal(base + gamma * b, 0.5 * np.diag(np.power(b, 3)))
        cap = quicksum(theta)
        val = c1 * cap + c2 * cap * cap - quicksum(theta[k] * r[k] for k in range(5))
        inner_sum += val
    model.addConstr(inner_sum/m - alpha <= u[i], "u" + str(i))

# model.write("out.lp")
# solve the model
model.optimize()
print("objective: ", model.objVal)
print("theta: ", theta)
# print("alpha: ", alpha)


