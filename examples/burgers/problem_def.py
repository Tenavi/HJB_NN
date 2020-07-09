import numpy as np
from utilities.other import cheb
from examples.problem_def_template import config_prototype, problem_prototype

class config_NN (config_prototype):
    def __init__(self, N_states, time_dependent):
        N_layers = 3
        N_neurons = 64
        self.layers = self.build_layers(N_states,
                                        time_dependent,
                                        N_layers,
                                        N_neurons)

        self.random_seeds = {'train': None}

        self.ODE_solver = 'BDF'
        # Accuracy level of BVP data
        self.data_tol = 1e-05
        # Max number of nodes to use in BVP
        self.max_nodes = 5000
        # Time horizon
        self.t1 = 8.

        # Time subintervals to use in time marching
        self.tseq = self.t1 * np.array([1e-03, 1e-02, .1, .125, .15,
                                        .2, .3, .4, .5, .6, .7, .8, .9, 1.])

        # Time step for integration and sampling
        self.dt = 1e-02
        # Standard deviation of measurement noise
        self.sigma = 1e-02

        # Which dimensions to plot when predicting value function V(0,x)?
        # (unspecified dimensions are held at mean value)
        self.plotdims = [0,9]

        # Number of training trajectories
        self.Ns = {'train': 30, 'val': 50}

        ##### Options for training #####

        # Number of data points to use in first training rounds
        # Set to None to use whole data set
        self.batch_size = None

        # Maximum factor to increase data set size each round
        self.Ns_scale = 2
        # Number of candidate points to pick from when selecting large gradient
        # points during adaptive sampling
        self.Ns_cand = 2
        # Maximum size of batch size to use
        self.Ns_max = 32768

        # Convergence tolerance parameter (see paper)
        self.conv_tol = 1e-03

        # maximum and minimum number of training rounds
        self.max_rounds = 1
        self.min_rounds = 1

        # List or array of weights on gradient term, length = max_rounds
        self.weight_A = [10.]
        # List or array of weights on control learning term, not used in paper
        self.weight_U = [0.]

        # Dictionary of lists or arrays of options to be passed to L-BFGS-B
        # The length of each list should be = max_rounds
        # Leave empty for default values
        self.BFGS_opts = {}

class setup_problem(problem_prototype):
    def __init__(self):
        self.N_states = 20
        self.N_controls = 1
        self.t1 = 8.

        self.nu = 0.2
        self.alpha = 1.5
        self.beta = -0.1
        self.omega = [-0.5, -0.2]
        self.W1 = 0.1
        self.W2 = 1.0

        # Chebyshev nodes, differentiation matrices, and Clenshaw-Curtis weights
        self.xi, self.D, self.w_flat = cheb(self.N_states+1)
        self.D2 = np.matmul(self.D, self.D)

        # Truncates system to account for zero boundary conditions
        self.xi = self.xi[1:-1]
        self.w_flat = self.w_flat[1:-1]
        self.w = self.w_flat.reshape(-1,1)
        self.D = self.D[1:-1, 1:-1]
        self.D2 = self.D2[1:-1, 1:-1]

        # Indicator function
        self.Iw_idx = (self.xi > self.omega[0]) & (self.xi < self.omega[1])
        self.Iw_vec_flat = self.Iw_idx.astype(np.float64)
        self.Iw_vec = self.Iw_vec_flat.reshape(-1,1)
        self.Iw_idx = np.arange(0, self.N_states)[self.Iw_idx]

        # Initial condition bounds
        self.X0_ub = np.full((self.N_states, 1), 2.)
        self.X0_lb = - self.X0_ub

        # Dynamics linearized around origin (dxdt ~= Fx + Gu)
        F = self.nu * self.D2 + self.alpha * np.identity(self.N_states)
        G = self.Iw_vec

        # Cost matrices
        Q = np.diag(self.w_flat / 2.)
        Rinv = np.array([[2./self.W1]])

        P1 = np.diag(self.w_flat * self.W2 / 2.)

        self.P = self.make_LQR(F, G, Q, Rinv, P1)
        self.RG = - Rinv @ G.T

    def U_star(self, X_aug):
        '''Control as a function of the costate.'''
        A = X_aug[self.N_states:2*self.N_states]
        U = - A[self.Iw_idx].sum(axis=0, keepdims=True) / self.W1
        return U

    def make_U_NN(self, A):
        '''Makes TensorFlow graph of optimal control with NN value gradient.'''
        from tensorflow import reduce_sum

        U = reduce_sum(A[self.Iw_idx[0]:self.Iw_idx[-1]+1],
                       axis=0, keepdims=True) / -self.W1

        return U

    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.N_states]
            XT = X_aug_T[:self.N_states]
            AT = X_aug_T[self.N_states:2*self.N_states]
            vT = X_aug_T[2*self.N_states:]

            # Derivative of the terminal cost with respect to X(T)
            dFdXT = self.W2 * XT * self.w_flat

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc

    def running_cost(self, X, U, wX=None):
        if wX is None:
            wX = self.w * X
        L = np.sum(np.sum(wX*X, axis=0, keepdims=True) + self.W1 * U**2,
                   axis=0, keepdims=True) / 2.

        return L

    def terminal_cost(self, X):
        return self.W2 / 2. * np.dot(X**2, self.w)

    def dynamics(self, t, X, U_fun):
        '''Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration.'''
        U = U_fun([[t]], X.reshape((-1,1))).flatten()

        dXdt = X * np.matmul(self.D, X) + self.nu * np.matmul(self.D2, X) \
            + self.alpha * X * np.exp(self.beta * X) + self.Iw_vec_flat * U

        return dXdt

    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP.'''

        # Control as a function of the costate
        U = self.U_star(X_aug)

        X = X_aug[:self.N_states]
        A = X_aug[self.N_states:2*self.N_states]

        wX = self.w * X
        DX = np.matmul(self.D, X)

        aeX = self.alpha * np.exp(self.beta * X)
        aXeX = X * aeX

        dXdt = X * DX + self.nu * np.matmul(self.D2, X) + \
               aXeX + self.Iw_vec * U.flatten()

        dAdt = - wX - np.matmul(self.D.T, X*A) - \
               self.nu * np.matmul(self.D2.T, A) - \
               (DX + self.beta*aXeX + aeX) * A

        L = self.running_cost(X_aug[:self.N_states], U, wX)

        return np.vstack((dXdt, dAdt, -L))
