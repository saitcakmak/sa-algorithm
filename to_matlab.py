import numpy as np
import scipy.io

data = np.load("output/quad_values_large_N_1000_runs_1000.npy").item()

out = dict()

prefix = 'l1000_val_'

out[prefix + 'empirical'] = data['empirical']
out[prefix + 'var05'] = data['var_0.5']
out[prefix + 'var07'] = data['var_0.7']
out[prefix + 'var09'] = data['var_0.9']
out[prefix + 'cvar05'] = data['cvar_0.5']
out[prefix + 'cvar07'] = data['cvar_0.7']
out[prefix + 'cvar09'] = data['cvar_0.9']

scipy.io.savemat("output/matlab/l1000_val.mat", out)
