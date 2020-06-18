'''
This script runs a closed-loop simulation where the controller is implemented
using a zero-order-hold (ZOH) and has measurement noise.
'''

import numpy as np
from scipy.integrate import solve_bvp
import scipy.io
import sys

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from utilities.neural_networks import hjb_network
    system += '/tspan'
else:
    from utilities.neural_networks import hjb_network_t0 as hjb_network
    system += '/t0'

# Loads pre-trained NN for control

parameters, scaling = load_NN('examples/' + system + '/V_model.mat')

model = hjb_network(problem, scaling, config, parameters)
model.run_initializer()

# Initializes some parameters

t1 = config.t1
N_states = problem.N_states

dt = config.dt
t = np.arange(0., t1+dt/2., dt)
Nt = t.shape[0]

if len(sys.argv) > 1:
    np.random.seed(int(sys.argv[1]))

# Generates the initial condition

if system[:7] == 'burgers':
    X0 = -2. * np.sin(problem.xi * np.pi)
else:
    X0 = problem.sample_X0(100)
    max_idx = model.get_largest_A(np.zeros((1, X0.shape[1])), X0, 1)
    X0 = X0[:, max_idx[-1]]
bc = problem.make_bc(X0)

# Samples Gaussian noise with variance defined in config

W = config.sigma * np.random.randn(N_states, Nt)

X_NN = np.empty((N_states, Nt))
X_LQR = np.empty((N_states, Nt))
X_NN[:,0] = X0
X_LQR[:,0] = X0

U_LQR = np.empty((problem.N_controls, Nt))
U_LQR[:,0] = problem.U_LQR(0., X0)

# ---------------------------------------------------------------------------- #

def RK4_step(f, t, X, U_fun, W):
    '''Simulates ZOH with measurement noise.'''
    U_eval = U_fun(t.reshape(1,1), np.reshape(X+W, (-1,1)))
    U = lambda t, X: U_eval

    # M = 4 steps of RK4 for each sample taken
    M = 4
    _dt = dt/M
    t0 = np.copy(t)
    X1 = np.copy(X)
    for _ in range(M):
        k1 = _dt * f(t, X, U)
        k2 = _dt * f(t + _dt/2., X + k1/2., U)
        k3 = _dt * f(t + _dt/2., X + k2/2., U)
        k4 = _dt * f(t + _dt, X + k3, U)

        X1 += (k1 + 2.*(k2 + k3) + k4)/6.
        t0 += _dt

    return X1

# Integrates the closed-loop system (NN & LQR controllers, RK4)

for k in range(1,Nt):
    X_NN[:,k] = RK4_step(
        problem.dynamics, t[k-1], X_NN[:,k-1], model.eval_U, W[:,k-1]
        )
    X_LQR[:,k] = RK4_step(
        problem.dynamics, t[k-1], X_LQR[:,k-1], problem.U_LQR, W[:,k-1]
        )

    U_LQR[:,k] = problem.U_LQR(t[k], X_LQR[:,k] + W[:,k])

V_NN, A_NN, U_NN = model.bvp_guess(t.reshape(1,-1), X_NN + W, eval_U=True)

save_dict = {'t': t, 'W': W, 'X_LQR': X_LQR, 'U_LQR': U_LQR,
             'X_NN': X_NN, 'A_NN': A_NN, 'V_NN': V_NN, 'U_NN': U_NN}

# Solves the two-point boundary value problem

SOL = solve_bvp(problem.aug_dynamics, bc, t, np.vstack((X_NN, A_NN, V_NN)),
                verbose=2, tol=config.data_tol, max_nodes=config.max_nodes)

X_aug = SOL.sol(save_dict['t'])

save_dict.update({'BVP_success': SOL.success,
                  'X_BVP': X_aug[:N_states],
                  'A_BVP': X_aug[N_states:2*N_states],
                  'V_BVP': X_aug[-1:],
                  'U_BVP': problem.U_star(X_aug),
                  'H_BVP': problem.Hamiltonian(save_dict['t'], X_aug)})

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
