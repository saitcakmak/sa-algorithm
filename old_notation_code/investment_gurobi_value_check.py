"""
this is an SAA implementation of investment problem based on v3
For reference in modeling, we use Qian2015Uncertainty-Framework pg 22-23
model (34) with CVaR replaced with expectation
"""
from gurobipy import *
from old_notation_code.investment_params import *
from theta_params import *


n, m = 10000, 1
delta = 0.9


# start gurobi model
model = Model("invest")
# actual decision variables
theta = model.addVars(range(5), name="theta", lb=-GRB.INFINITY)
# short sell amount variables
theta_minus = model.addVars(range(5), name="theta_minus", lb=0)
# the CVaR decision variable
alpha = model.addVar(name="alpha", lb=-GRB.INFINITY)
# auxiliary decision variables for each CVaR sample
u = model.addVars(range(n), lb=0, name="u")
# set objective value
model.setObjective(alpha + (1/((1 - delta) * n)) * quicksum(u), GRB.MINIMIZE)

# constraints:
cap = quicksum(theta)
short = quicksum(theta_minus)
model.addConstrs((-theta[i] <= theta_minus[i] for i in range(5)), name="short_val")
for i in range(n):
    print("iter: ", i)
    inner_sum = 0
    gamma = np.random.randn() * std_gamma + mu_gamma
    # generate inner expectation
    for j in range(m):
        r = base + gamma * b
        val = c1 * cap + c2 * cap * cap + c3 * short - quicksum(theta[k] * r[k] for k in range(5))
        inner_sum += val
    model.addConstr(inner_sum/m - alpha <= u[i], "u" + str(i))

# model.write("out.lp")
# solve the model
model.optimize()
t_return = []
t_m_return = []
for i in range(5):
    t_return.append(theta[i].X)
    t_m_return.append(theta_minus[i].X)
print("objective: ", model.objVal)
print("theta: ", t_return)
print("theta_minus: ", t_m_return)
print("alpha: ", alpha.X)


