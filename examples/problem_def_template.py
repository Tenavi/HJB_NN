'''
This script provides common functions used to help define optimal control
problems, as well as prototype classes to base new problems on.
'''

import numpy as np
from scipy.integrate import cumtrapz, solve_ivp

class config_prototype:
    def __init__(self, N_states, t1, time_dependent):
        '''Class defining problem and training parameters.'''
        N_layers = 3
        N_neurons = 64
        self.layers = self.build_layers(N_states,
                                        time_dependent,
                                        N_layers,
                                        N_neurons)

        self.random_seeds = {'train': None}

        self.ODE_solver = 'DOP853'
        # Accuracy level of BVP data
        self.data_tol = 1e-05
        # Max number of nodes to use in BVP
        self.max_nodes = 5000

        # Time subintervals to use in time marching
        Nt = 8
        self.tseq = np.linspace(0., t1, Nt+1)[1:]

        # Time step for integration and sampling
        self.dt = 1e-02
        # Standard deviation of measurement noise
        self.sigma = 1e-02

        # Which dimensions to plot when predicting value function V(0,x)?
        # (unspecified dimensions are held at mean value)
        self.plotdims = [0,1]

        # Number of training trajectories
        self.Ns = {'train': 64, 'val': 1000}

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
        # Portion of data set size to use after the first round
        self.Ns_sub_size = 1/2

        # Convergence tolerance parameter (see paper)
        self.conv_tol = 2.*1e-03

        # maximum and minimum number of training rounds
        self.max_rounds = 1
        self.min_rounds = 1

        # List or array of weights on gradient term, length = max_rounds
        self.weight_A = [1.]*self.max_rounds
        # List or array of weights on control learning term, not used in paper
        self.weight_U = [0.]*self.max_rounds

        # Dictionary of lists or arrays of options to be passed to L-BFGS-B
        # The length of each list should be = max_rounds
        # Leave empty for default values
        self.BFGS_opts = {}

    def build_layers(self, N_states, time_dependent, N_layers, N_neurons):
        layers = [N_states] + N_layers * [N_neurons] + [1]

        if time_dependent:
            layers[0] += 1

        return layers

class problem_prototype:
    def __init__(self):
        '''Class defining the OCP, dynamics, and controllers.'''
        raise NotImplementedError

    def sample_X0(self, Ns):
        '''Uniform sampling from the initial condition domain.'''
        X0 = np.random.rand(self.N_states, Ns)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def U_star(self, X_aug):
        '''Optimal control as a function of the costate.'''
        raise NotImplementedError

    def make_U_NN(self, A):
        '''Makes TensorFlow graph of optimal control with NN value gradient.'''
        import tensorflow as tf

        raise NotImplementedError

    def make_LQR(self, F, G, Q, Rinv, P1):
        '''Solves the Riccati ODE for this OCP.'''
        GRG = G @ Rinv @ G.T
        def riccati_ODE(t, p):
            P = p.reshape(F.shape)
            PF = - P @ F
            dPdt = PF.T + PF - Q + P @ GRG @ P
            return dPdt.flatten()
        SOL = solve_ivp(riccati_ODE, [t1, 0.], P1.flatten(),
            dense_output=True, method='LSODA', rtol=1e-04)
        return SOL.sol

    def U_LQR(self, t, X):
        '''Evaluates the LQR controller for this OCP.'''
        t = np.reshape(t, (1,))
        P = self.P(t).reshape(self.N_states, self.N_states)
        return self.RG @ P @ X

    def make_bc(self, X0_in):
        '''Makes a function to evaluate the boundary conditions for a given
        initial condition.
        (terminal cost is zero so final condition on lambda is zero)'''
        def bc(X_aug_0, X_aug_T):
            return np.concatenate((X_aug_0[:self.N_states] - X0_in,
                                   X_aug_T[self.N_states:]))
        return bc

    def running_cost(self, X, U):
        raise NotImplementedError

    def terminal_cost(self, X):
        raise NotImplementedError

    def compute_cost(self, t, X, U):
        '''Computes the accumulated cost of a state-control trajectory as
        an approximation of V(t).'''
        L = self.running_cost(X, U)
        J = cumtrapz(L, t, initial=0.)
        return self.terminal_cost(X[:,-1]) + J[0,-1] + L[0,-1] - J

    def Hamiltonian(self, t, X_aug):
        '''Evaluates the Hamiltonian for this OCP.'''
        U = self.U_star(X_aug)
        L = self.running_cost(X_aug[:self.N_states], U)

        F = self.aug_dynamics(t, X_aug)
        H = L + np.sum(X_aug[self.N_states:2*self.N_states] * F[:self.N_states],
            axis=0, keepdims=True)

        return H

    def dynamics(self, t, X, U_fun):
        '''Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration.'''
        U = U_fun([[t]], X.reshape((-1,1))).flatten()

        raise NotImplementedError

    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP.'''

        # Optimal control as a function of the costate
        U = self.U_star(X_aug)

        raise NotImplementedError
