import numpy as np
import scipy.stats
from scipy.integrate import solve_bvp
import scipy.io
import time
import warnings

from utilities.other import int_input
from examples.choose_problem import system, problem, config, time_dependent

X0_pool = scipy.io.loadmat('examples/' + system + '/X0_pool.mat')['X0']

np.seterr(over='warn', divide='warn', invalid='warn')
warnings.filterwarnings('error')

N_states = problem.N_states
Nt = config.tseq.shape[0]
Ns = X0_pool.shape[1]

N_converged = 0
avg_time = []

print('')
print('Testing time marching with Nt =', Nt, 'time intervals...')
print('')

# ---------------------------------------------------------------------------- #

for i in range(Ns):
    print('Solving BVP #', i+1, 'of', Ns, '...', end='\r')

    X0 = X0_pool[:,i]
    bc = problem.make_bc(X0)

    start_time = time.time()

    try:
        # Initial guess is zeros
        t_guess = np.array([0.])
        X_guess = np.vstack((X0.reshape(-1,1),
                             np.zeros((N_states+1, 1))))

        tol = 1e-01

        ##### Time-marching to build from t0 to tf #####
        for k in range(Nt):
            if tol >= 2.*config.data_tol:
                tol /= 2.
            if k == Nt - 1:
                tol = config.data_tol

            t_guess = np.concatenate((t_guess, config.tseq[k:k+1]))
            X_guess = np.hstack((X_guess, X_guess[:,-1:]))

            SOL = solve_bvp(problem.aug_dynamics, bc, t_guess, X_guess,
                            tol=tol, max_nodes=config.max_nodes)

            if not SOL.success:
                warnings.warn(Warning())
            t_guess = SOL.x
            X_guess = SOL.y

        avg_time.append(time.time() - start_time)
        N_converged += 1

    except Warning:
        pass

print('')
print(N_converged, '/', Ns, 'successful solution attempts:')
print('Mean solution time: %1.1f' % (np.mean(avg_time)), 'sec')
