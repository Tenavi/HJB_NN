'''
This script predicts the value function at initial time on a grid.
'''

import numpy as np
import scipy.io
import time
import sys

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from utilities.neural_networks import hjb_network
    system += '/tspan'
else:
    from utilities.neural_networks import hjb_network_t0 as hjb_network
    system += '/t0'

parameters, scaling = load_NN('examples/' + system + '/V_model.mat')

model = hjb_network(problem, scaling, config, parameters)

# ---------------------------------------------------------------------------- #

pred_time = time.time()

ub = problem.X0_ub
lb = problem.X0_lb

plotdims = config.plotdims
Nm = [100,100]

Nout = np.prod(Nm)

# Plots mean value of x if not part of plotdims
X = np.tile((ub + lb)/2., (1, Nout))

# Makes a meshgrid out of plotdims
X_mesh = []
for d in range(len(plotdims)):
    X_mesh.append(np.linspace(lb[plotdims[d]], ub[plotdims[d]], Nm[d]))
X_mesh = np.meshgrid(*X_mesh)

for d in range(len(plotdims)):
    X[plotdims[d],:] = X_mesh[d].flatten()

V = model.predict_V(np.zeros((1, Nout)), X).reshape(Nm)

pred_time = time.time() - pred_time

print('Prediction time: %.1f' % (pred_time))

save_dict = {'plotdims': np.array(plotdims)+1, 'X': X_mesh, 'V': V,
             'U': model.eval_U(np.zeros((1, Nout)), X)[0].reshape(Nm)}
scipy.io.savemat('examples/' + system + '/results/val_pred.mat', save_dict)
