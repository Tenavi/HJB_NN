'''
This script tests the NN warm start method for solving BVPs.
It uses the initial conditions provided in examples/<system>/X0_pool.mat.
'''

import numpy as np
import scipy.stats
from scipy.integrate import solve_ivp, solve_bvp
import scipy.io
import time
import warnings

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

X0_pool = scipy.io.loadmat('examples/' + system + '/X0_pool.mat')['X0']

if time_dependent:
    from utilities.neural_networks import hjb_network
    system += '/tspan'
else:
    from utilities.neural_networks import hjb_network_t0 as hjb_network
    system += '/t0'

# Loads the pre-trained NN

parameters, scaling, NN_stats = load_NN(
    'examples/' + system + '/V_model.mat', return_stats=True)
train_time, val_grad_err, val_ctrl_err = NN_stats

model = hjb_network(problem, scaling, config, parameters)
model.run_initializer()

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

N_states = problem.N_states
Ns = X0_pool.shape[1]

N_converged = 0
avg_time = []

print('')
print('Testing NN warm start:')
print('NN was trained in %.0f' % (train_time), 'sec')
print('NN mean relative costate prediction error = %1.1e' % (val_grad_err))
print('NN mean relative control prediction error = %1.1e' % (val_ctrl_err))
print('')

# ---------------------------------------------------------------------------- #

for i in range(Ns):
    print('Solving BVP #', i+1, 'of', Ns, '...', end='\r')

    X0 = X0_pool[:,i]
    bc = problem.make_bc(X0)

    start_time = time.time()

    try:
        ##### Integrates the closed-loop system (NN controller) #####

        SOL = solve_ivp(problem.dynamics, [0., config.t1], X0,
                        method=config.ODE_solver,
                        args=(model.eval_U,),
                        rtol=1e-04)

        V_guess, A_guess = model.bvp_guess(SOL.t.reshape(1,-1), SOL.y)

        ##### Solves the two-point boundary value problem #####

        X_aug_guess = np.vstack((SOL.y, A_guess, V_guess))
        SOL = solve_bvp(problem.aug_dynamics, bc, SOL.t, X_aug_guess,
                        tol=config.data_tol,
                        max_nodes=config.max_nodes)

        if not SOL.success:
            warnings.warn(Warning())

        avg_time.append(time.time() - start_time)
        N_converged += 1

    except Warning:
        pass

print('')
print(N_converged, '/', Ns, 'successful solution attempts:')
print('Mean solution time: %1.1f' % (np.mean(avg_time)), 'sec')
