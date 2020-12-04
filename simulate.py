'''
This script runs a closed-loop simulation where the controller is continuous
and is allowed continuous, noise-free measurements.
'''

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import scipy.io
import time
import sys

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from utilities.neural_networks import HJBnet
    system += '/tspan'
else:
    from utilities.neural_networks import HJBnet_t0 as HJBnet
    system += '/t0'

# Loads pre-trained NN for control

parameters, scaling = load_NN('examples/' + system + '/V_model.mat')

model = HJBnet(problem, scaling, config, parameters)

# Initializes some parameters

t1 = problem.t1
N_states = problem.N_states

if len(sys.argv) > 1:
    np.random.seed(int(sys.argv[1]))

# Generates the initial condition

if system[:7] == 'burgers':
    X0 = -2. * np.sin(problem.xi * np.pi)
else:
    X0 = problem.sample_X0(1)
bc = problem.make_bc(X0)

# ---------------------------------------------------------------------------- #

# Integrates the closed-loop system (NN controller)

start_time = time.time()

SOL = solve_ivp(problem.dynamics, [0., t1], X0,
                method=config.ODE_solver, args=(model.eval_U,), rtol=1e-04)

V_NN, A_NN, U_NN = model.bvp_guess(SOL.t.reshape(1,-1), SOL.y, eval_U=True)

save_dict = {'NN_time': time.time() - start_time, 't': SOL.t,
             'X_NN': SOL.y, 'A_NN': A_NN, 'V_NN': V_NN, 'U_NN': U_NN}

# Solves the two-point boundary value problem

start_time = time.time()

SOL = solve_bvp(problem.aug_dynamics, bc, SOL.t, np.vstack((SOL.y, A_NN, V_NN)),
                verbose=2, tol=config.data_tol, max_nodes=config.max_nodes)

X_aug = SOL.sol(save_dict['t'])

save_dict.update({'BVP_success': SOL.success,
                  'BVP_time': time.time() - start_time,
                  'X_BVP': X_aug[:N_states],
                  'A_BVP': X_aug[N_states:2*N_states],
                  'V_BVP': X_aug[-1:],
                  'U_BVP': problem.U_star(X_aug),
                  'H_BVP': problem.Hamiltonian(save_dict['t'], X_aug)})

# Integrates the closed-loop system (LQR controller)

start_time = time.time()

SOL = solve_ivp(problem.dynamics, [0., t1], X0,
                method=config.ODE_solver, args=(problem.U_LQR,), rtol=1e-04,
                dense_output=True)

X = SOL.sol(save_dict['t'])
U = np.empty((problem.N_controls, X.shape[1]))
for k in range(U.shape[1]):
    U[:,k] = problem.U_LQR(save_dict['t'][k], X[:,k])

save_dict.update({'LQR_time': time.time() - start_time, 'X_LQR': X, 'U_LQR': U})

# ---------------------------------------------------------------------------- #

NN_cost = problem.compute_cost(
    save_dict['t'], save_dict['X_NN'], save_dict['U_NN'])[0,0]
LQR_cost = problem.compute_cost(
    save_dict['t'], save_dict['X_LQR'], save_dict['U_LQR'])[0,0]

print('')
print('NN cost: %.2f' % (NN_cost),
    ' (%.2f' % (100.*(NN_cost/save_dict['V_BVP'][0,0] - 1.)), '% suboptimal)')
print('LQR cost: %.2f' % (LQR_cost),
    ' (%.2f' % (100.*(LQR_cost/save_dict['V_BVP'][0,0] - 1.)), '% suboptimal)')
print('Optimal cost: %.2f' % (save_dict['V_BVP'][0,0]))

try:
    save_dict.update({'xi': problem.xi})
except:
    pass

scipy.io.savemat('examples/' + system + '/results/sim_data.mat', save_dict)
