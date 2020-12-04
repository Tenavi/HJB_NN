'''
This script generates data from scratch using time-marching.
'''

import numpy as np
import scipy.stats
from scipy.integrate import solve_bvp
import scipy.io
import time
import warnings

from utilities.other import int_input

from examples.choose_problem import system, problem, config

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

N_states = problem.N_states

# Testing or training data?
data_type = int_input('What kind of data? Enter 0 for test, 1 for training:')
if data_type:
    data_type = 'train'
else:
    data_type = 'test'

Ns = config.Ns[data_type]
X0_pool = problem.sample_X0(Ns)

# Arrays to store generated data
t_OUT = np.empty((1,0))
X_OUT = np.empty((N_states,0))
A_OUT = np.empty((N_states,0))
V_OUT = np.empty((1,0))

N_sol = 0
N_fail = 0
sol_time = []
fail_time = []

# ---------------------------------------------------------------------------- #

while N_sol < Ns:
    print('Solving BVP #', N_sol+1, 'of', Ns, '...', end='\r')

    X0 = X0_pool[:,N_sol]
    bc = problem.make_bc(X0)

    start_time = time.time()

    try:
        # Initial guess is zeros
        t_guess = np.array([0.])
        X_guess = np.vstack((X0.reshape(-1,1),
                             np.zeros((N_states+1, 1))))

        tol = 1e-01

        ##### Time-marching to build from t0 to tf #####
        for k in range(config.tseq.shape[0]):
            if tol >= 2.*config.data_tol:
                tol /= 2.
            if k == config.tseq.shape[0] - 1:
                tol = config.data_tol

            t_guess = np.concatenate((t_guess, config.tseq[k:k+1]))
            X_guess = np.hstack((X_guess, X_guess[:,-1:]))

            SOL = solve_bvp(problem.aug_dynamics, bc, t_guess, X_guess,
                            verbose=0, tol=tol, max_nodes=config.max_nodes)

            if not SOL.success:
                print(SOL.message)
                warnings.warn(Warning())
            t_guess = SOL.x
            X_guess = SOL.y

        sol_time.append(time.time() - start_time)

        V = SOL.y[-1:] + problem.terminal_cost(SOL.y[:N_states,-1])

        t_OUT = np.hstack((t_OUT, SOL.x.reshape(1,-1)))
        X_OUT = np.hstack((X_OUT, SOL.y[:N_states]))
        A_OUT = np.hstack((A_OUT, SOL.y[N_states:2*N_states]))
        V_OUT = np.hstack((V_OUT, V))

        N_sol += 1

    except Warning:
        fail_time.append(time.time() - start_time)

        X0_pool[:,N_sol] = problem.sample_X0(1)

        N_fail += 1

# ---------------------------------------------------------------------------- #

sol_time = np.sum(sol_time)
fail_time = np.sum(fail_time)

print('')
print(N_sol, '/', N_sol + N_fail, 'successful solution attempts:')
print('Average solution time: %1.1f' % (sol_time/N_sol), 'sec')
print('Total solution time: %1.1f' % (sol_time), 'sec')
if N_fail >= 1:
    print('')
    print('Average failure time: %1.1f' % (fail_time/(N_fail)), 'sec')
    print('Total failure time: %1.1f' % (fail_time), 'sec')
    print('Total working time: %1.1f' % (sol_time + fail_time), 'sec')

print('')
print('Total data generated:', X_OUT.shape[1])
print('')

# ---------------------------------------------------------------------------- #

save_data = int_input('Save data? Enter 0 for no, 1 for yes:')

if save_data:
    save_path = 'examples/' + system + '/data_' + data_type + '.mat'

    try:
        save_dict = scipy.io.loadmat(save_path)

        overwrite_data = int_input('Overwrite existing data? Enter 0 for no, 1 for yes:')

        if overwrite_data:
            raise

        save_dict.update({'t': np.hstack((save_dict['t'], t_OUT)),
                          'X': np.hstack((save_dict['X'], X_OUT)),
                          'A': np.hstack((save_dict['A'], A_OUT)),
                          'V': np.hstack((save_dict['V'], V_OUT))})
    except:
        save_dict = {'t': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT}
    scipy.io.savemat(save_path, save_dict)
