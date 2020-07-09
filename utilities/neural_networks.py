'''
This script contains the classes implementing neural networks modeling V(t,x)
and V(0,x). These are hjb_network and hjb_network_t0, respectively.
'''

import numpy as np
import tensorflow as tf
from scipy import stats
from scipy.integrate import solve_ivp, solve_bvp
import time

class hjb_network:
    def __init__(self, problem, scaling, config, parameters=None):
        '''Class implementing a NN for modeling time-dependent value functions.
        problem: instance of a problem class
        scaling: dictionary with 8 components:
            'lb' and 'ub',
                the lower and upper bounds of the input data, prior to scaling
            'A_lb' and 'A_ub',
                the lower and upper bounds of the gradient data, prior to scaling
            'U_lb' and 'U_ub',
                the lower and upper bounds of the control data, prior to scaling
            'V_min', and 'V_max',
                the lower and upper bounds of the output data, prior to scaling
        config: config_NN instance
        parameters: dict of weights and biases with pre-trained weights and biases'''

        self.lb = scaling['lb']
        self.ub = scaling['ub']
        self.A_lb = scaling['A_lb']
        self.A_ub = scaling['A_ub']
        self.U_lb = scaling['U_lb']
        self.U_ub = scaling['U_ub']
        self.V_min = scaling['V_min']
        self.V_max = scaling['V_max']

        self.problem = problem
        self.config = config

        N_states = problem.N_states
        N_controls = problem.N_controls
        self.t1 = config.t1

        # Initializes the NN parameters
        self.weights, self.biases = self.initialize_net(config.layers, parameters)

        # Defines placeholders for passing inputs and data
        self.t_tf = tf.placeholder(tf.float32, shape=(1, None))
        self.X_tf = tf.placeholder(tf.float32, shape=(N_states, None))
        self.A_tf = tf.placeholder(tf.float32, shape=(N_states, None))
        self.U_tf = tf.placeholder(tf.float32, shape=(N_controls, None))
        self.V_tf = tf.placeholder(tf.float32, shape=(1, None))

        self.A_scaled_tf = tf.placeholder(tf.float32, shape=(N_states, None))
        self.U_scaled_tf = tf.placeholder(tf.float32, shape=(N_controls, None))
        self.V_scaled_tf = tf.placeholder(tf.float32, shape=(1, None))

        # Builds the computational graph
        V_pred_scaled, self.V_pred = self.make_eval_graph(self.t_tf, self.X_tf)
        self.dVdX = tf.gradients(self.V_pred, self.X_tf)[0]
        self.U = self.problem.make_U_NN(self.dVdX)

        dVdX_norm = tf.sqrt(tf.reduce_sum(self.dVdX**2, axis=0))
        self.k_largest = tf.placeholder(tf.int32, ())
        self.largest_dVdX = tf.nn.top_k(dVdX_norm, k=self.k_largest, sorted=False)

        # Unweighted MSE loss on scaled data
        self.loss_V = tf.reduce_mean((V_pred_scaled - self.V_scaled_tf)**2)

        # Unweighted MSE loss on value gradient
        dVdX_scaled = 2.0*(self.dVdX - self.A_lb)/(self.A_ub - self.A_lb) - 1.0
        self.loss_A = tf.reduce_mean(
            tf.reduce_sum((dVdX_scaled - self.A_scaled_tf)**2, axis=0)
            )

        # Control loss
        U_scaled = 2.0*(self.U - self.U_lb)/(self.U_ub - self.U_lb) - 1.0
        self.loss_U = tf.reduce_mean(
            tf.reduce_sum((U_scaled - self.U_scaled_tf)**2, axis=0)
            )

        # Error metrics
        self.MAE = tf.reduce_mean(tf.abs(self.V_pred - self.V_tf))
        self.grad_MRL2 = tf.reduce_mean(tf.sqrt(
            tf.reduce_sum((self.dVdX - self.A_tf)**2, axis=0) / (
            0.01 + tf.reduce_sum(self.A_tf**2, axis=0)))
            )
        self.ctrl_MRL2 = tf.reduce_mean(tf.sqrt(
            tf.reduce_sum((self.U - self.U_tf)**2, axis=0) / (
            0.01 + tf.reduce_sum(self.U_tf**2, axis=0)))
            )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def initialize_net(self, layers, parameters):
        '''
        Creates Tensorflow variables for NN parameters. These are initialized
        with existing values if provided in the parameters argument.
        If not provided, then they are initialized with Xavier initialization.
        '''
        weights, biases = [], []
        if parameters is None:
            def xavier_init(size_in, size_out):
                # Initializes a single set of weights for layer (l) from layer (l-1).
                # Weights are picked randomly from a normal distribution
                std = np.sqrt(2. / (size_in + size_out))
                init = std * np.random.randn(size_out, size_in)
                return tf.Variable(init, dtype=tf.float32)

            for l in range(len(layers) - 1):
                weights.append(xavier_init(layers[l], layers[l+1]))
                biases.append(tf.Variable(tf.zeros((layers[l+1], 1), dtype=tf.float32)))
        else:
            for l in range(len(parameters['weights'])):
                weights.append(tf.Variable(parameters['weights'][l], dtype=tf.float32))
                biases.append(tf.Variable(parameters['biases'][l], dtype=tf.float32))

        return weights, biases

    def export_model(self):
        '''Returns a list of weights and biases to save model parameters.'''
        weights = np.empty((len(self.weights),), dtype=object)
        biases = np.empty((len(self.biases),), dtype=object)

        for l in range(len(self.weights)):
            weights[l], biases[l] = self.sess.run(
                (self.weights[l], self.biases[l])
                )

        return weights, biases

    def make_eval_graph(self, t, X):
        '''Builds the NN computational graph.'''

        # (N_states, ?) matrix of linearly rescaled input values
        V = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        V = tf.concat([V, 2.0*t/self.t1 - 1.0], axis=0)
        # Hidden layers
        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            V = tf.tanh(tf.matmul(W, V) + b)
        # The last layer is linear -> it's outside the loop
        W = self.weights[-1]
        b = self.biases[-1]
        V = tf.matmul(W, V) + b

        V_descaled = (self.V_max - self.V_min)*(V + 1.)/2. + self.V_min

        return V, V_descaled

    def predict_V(self, t, X):
        '''Run a TensorFlow Session to predict the value function at arbitrary
        space-time coordinates.'''
        return self.sess.run(self.V_pred, {self.t_tf: t, self.X_tf: X})

    def predict_A(self, t, X):
        '''Run a TensorFlow Session to predict the value gradient at arbitrary
        space-time coordinates.'''
        return self.sess.run(self.dVdX, {self.t_tf: t, self.X_tf: X})

    def get_largest_A(self, t, X, N):
        '''Partially sorts space-time points by the predicted gradient norm.'''
        _, max_idx = self.sess.run(self.largest_dVdX,
                                   {self.k_largest: N,
                                    self.t_tf: t,
                                    self.X_tf: X})
        return max_idx

    def eval_U(self, t, X):
        '''(Near-)optimal feedback control for arbitrary inputs (t,X).'''
        return self.sess.run(self.U, {self.t_tf: t, self.X_tf: X}).astype(np.float64)

    def bvp_guess(self, t, X, eval_U=False):
        '''Predicts value, costate, and control with one session call.'''
        feed_dict = {self.t_tf: t, self.X_tf: X}

        if eval_U:
            V, A, U = self.sess.run((self.V_pred, self.dVdX, self.U), feed_dict)
            return V, A, U.astype(np.float64)
        else:
            return self.sess.run((self.V_pred, self.dVdX), feed_dict)

    def train(self, train_data, val_data):
        '''Implements training with L-BFGS.'''

        train_data.update({
            'U': self.problem.U_star(np.vstack((train_data['X'], train_data['A'])))
            })
        train_data.update({
            'A_scaled': 2.*(train_data['A'] - self.A_lb)/(self.A_ub - self.A_lb) - 1.,
            'U_scaled': 2.*(train_data['U'] - self.U_lb)/(self.U_ub - self.U_lb) - 1.,
            'V_scaled': 2.*(train_data['V'] - self.V_min)/(self.V_max - self.V_min) - 1.
            })

        val_data = {self.t_tf: val_data.pop('t'), self.X_tf: val_data.pop('X'),
                    self.V_tf: val_data.pop('V'), self.A_tf: val_data.pop('A')}
        val_data.update({
            self.U_tf: self.problem.U_star(
                np.vstack((val_data[self.X_tf], val_data[self.A_tf]))
                )
            })

        # ----------------------------------------------------------------------

        # Gets training options from configuration file
        self.Ns = self.config.batch_size
        if self.Ns is None:
            self.Ns = train_data['X'].shape[1]

        Ns_C = self.config.Ns_scale
        Ns_cand = self.config.Ns_cand
        Ns_max = self.config.Ns_max

        conv_tol = self.config.conv_tol

        max_rounds = self.config.max_rounds
        min_rounds = self.config.min_rounds

        weight_A = self.config.weight_A
        self.weight_A_tf = tf.placeholder(tf.float32, shape=())

        weight_U = self.config.weight_U
        self.weight_U_tf = tf.placeholder(tf.float32, shape=())

        self.loss = self.loss_V
        if weight_A[0] >= 10.0 * np.finfo(float).eps:
            self.loss = self.loss + self.weight_A_tf * self.loss_A
        if weight_U[0] >= 10.0 * np.finfo(float).eps:
            self.loss = self.loss + self.weight_U_tf * self.loss_U

        # ----------------------------------------------------------------------

        # Sets up optimizer stuff
        self.grads_list = [None]*3
        optimizer = None

        train_err = []
        train_grad_err = []
        train_ctrl_err = []

        val_err = []
        val_grad_err = []
        val_ctrl_err = []

        round_iters = []

        errors_to_track = [train_err, train_grad_err, train_ctrl_err]
        fetches = [[self.MAE, self.grad_MRL2, self.ctrl_MRL2]]

        # ----------------------------------------------------------------------

        for round in range(1,max_rounds+1):
            # Generates new data if needed
            if self.Ns > train_data['X'].shape[1]:
                new_data = self.generate_data(
                    self.Ns - train_data['X'].shape[1], Ns_cand)
                for key in new_data.keys():
                    train_data.update({
                        key: np.hstack((train_data[key], new_data[key]))
                        })

            self.Ns = np.minimum(self.Ns, Ns_max)

            print('Optimization round', round, ':')
            print('Batch size =', self.Ns,
                  ', gradient weight = %1.1e' % (weight_A[round-1]),
                  ', control weight = %1.1e' % (weight_U[round-1]))

            # ------------------------------------------------------------------

            idx = np.random.choice(
                train_data['X'].shape[1], self.Ns, replace=False)

            tf_dict = {self.t_tf: train_data['t'][:,idx],
                       self.X_tf: train_data['X'][:,idx],
                       self.A_tf: train_data['A'][:,idx],
                       self.U_tf: train_data['U'][:,idx],
                       self.V_tf: train_data['V'][:,idx],
                       self.A_scaled_tf: train_data['A_scaled'][:,idx],
                       self.U_scaled_tf: train_data['U_scaled'][:,idx],
                       self.V_scaled_tf: train_data['V_scaled'][:,idx],
                       self.weight_A_tf: weight_A[round-1],
                       self.weight_U_tf: weight_U[round-1]}

            BFGS_opts = {}
            for key in self.config.BFGS_opts.keys():
                BFGS_opts[key] = self.config.BFGS_opts[key][round-1]

            optimizer = self._train_L_BFGS_B(tf_dict,
                optimizer=optimizer,
                errors_to_track=errors_to_track,
                fetches=fetches,
                options=BFGS_opts)

            # ------------------------------------------------------------------

            loss_V, loss_A, loss_U = self.sess.run(
                (self.loss_V, self.loss_A, self.loss_U), tf_dict)
            print('')
            print('loss_V = %1.1e' % (loss_V),
                  ', loss_A = %1.1e' % (loss_A),
                  ', loss_U = %1.1e' % (loss_U))

            # If didn't track training errors, compute them now
            for error,fetch in zip((train_err,train_grad_err,train_ctrl_err),
                               (self.MRAE, self.grad_MRL2, self.ctrl_MRL2)):
                if error not in errors_to_track:
                    error.append(self.sess.run(fetch, tf_dict))

            round_iters.append(len(train_err))

            val_errs = self.sess.run(
                (self.MAE, self.grad_MRL2, self.ctrl_MRL2), val_data)

            val_err.append(val_errs[0])
            val_grad_err.append(val_errs[1])
            val_ctrl_err.append(val_errs[2])

            print('')
            print('Training MAE error = %1.1e' % (train_err[-1]))
            print('Validation MAE error = %1.1e' % (val_err[-1]))
            print('Training grad. MRL2 error = %1.1e' % (train_grad_err[-1]))
            print('Validation grad. MRL2 error = %1.1e' % (val_grad_err[-1]))
            print('Training ctrl. MRL2 error = %1.1e' % (train_ctrl_err[-1]))
            print('Validation ctrl. MRL2 error = %1.1e' % (val_ctrl_err[-1]))

            # ------------------------------------------------------------------

            if max_rounds > 1:
                if self.convergence_test(tf_dict,
                                         optimizer.grad_eval,
                                         epsilon=conv_tol,
                                         Ns_sub=int(self.Ns/8),
                                         C=Ns_C):
                    if round >= min_rounds:
                        print('Convergence test satisfied, stopping optimization.')
                        break
                    else:
                        print('Convergence test satisfied, but have not trained for minimum number of rounds.')
                        self.Ns *= Ns_C

        errors = (train_err, train_grad_err, train_ctrl_err,
                  val_err, val_grad_err, val_ctrl_err)
        return round_iters, errors

    def _train_L_BFGS_B(self,
                        tf_dict,
                        optimizer=None,
                        errors_to_track=[],
                        fetches=[],
                        options={}):
        '''
        Interface for L-BFGS optimizer. Allows reusing an instantiated
        optimizer so that gradients etc. do not have to be recomputed each
        training round.
        '''
        from utilities.optimize import ScipyOptimizerInterface

        # Minimizes with L-BFGS
        if optimizer is None:
            default_opts = {'maxcor': 15, 'ftol': 1e-11, 'gtol': 1e-06,
                            'iprint': 95, 'maxfun': 100000, 'maxiter': 100000}

            optimizer = ScipyOptimizerInterface(self.loss,
                grads_list=self.grads_list,
                options={**default_opts, **options})
            self.grads_list = optimizer._grads_list
            self.packed_loss_grad = optimizer._packed_loss_grad

        def callback(fetches):
            for error_list, fetch in zip(errors_to_track, fetches):
                error_list.append(fetch)

        optimizer.minimize(self.sess, feed_dict=tf_dict,
                           fetches=fetches, loss_callback=callback)

        return optimizer

    def convergence_test(self, tf_dict, sample_grad, epsilon, Ns_sub=1024, C=2):
        '''Convergence test as described in the paper.'''
        print('')
        print('Running convergence test...')

        sample_grad = np.linalg.norm(sample_grad, ord=1)

        # Calculates sample variance for (a subsample of) the batch
        idx = np.random.choice(self.Ns, Ns_sub, replace=False)

        tf_dict.update({
            self.t_tf: tf_dict[self.t_tf][:,idx],
            self.X_tf: tf_dict[self.X_tf][:,idx],
            self.A_scaled_tf: tf_dict[self.A_scaled_tf][:,idx],
            self.U_scaled_tf: tf_dict[self.U_scaled_tf][:,idx],
            self.V_scaled_tf: tf_dict[self.V_scaled_tf][:,idx],
            })

        # ----------------------------------------------------------------------

        sample_var = np.empty((Ns_sub, self.packed_loss_grad.shape[0]))
        for i in range(Ns_sub):
            sample_var[i] = self.sess.run(self.packed_loss_grad, {
                self.t_tf: tf_dict[self.t_tf][:,i:i+1],
                self.X_tf: tf_dict[self.X_tf][:,i:i+1],
                self.A_scaled_tf: tf_dict[self.A_scaled_tf][:,i:i+1],
                self.U_scaled_tf: tf_dict[self.U_scaled_tf][:,i:i+1],
                self.V_scaled_tf: tf_dict[self.V_scaled_tf][:,i:i+1],
                self.weight_A_tf: tf_dict[self.weight_A_tf],
                self.weight_U_tf: tf_dict[self.weight_U_tf]
                })

        sample_var = np.var(sample_var, axis=0, ddof=1, dtype=np.float64)
        sample_var = sample_var.sum()

        # ----------------------------------------------------------------------

        print('sample variance = %1.4e' % (sample_var))
        print('sample gradient = %1.4e' % (sample_grad))
        print('epsilon = %1.4e' % (epsilon), ', Ns_sub =', Ns_sub)

        if sample_var <= epsilon * Ns_sub * sample_grad:
            # Convergence condition satisfied
            return True
        else:
            Ns_min = int(sample_var / (epsilon * sample_grad))
            if self.Ns >= Ns_min:
                Ns_min = int(self.Ns * (Ns_min / Ns_sub))
            self.Ns = np.minimum(C*self.Ns, Ns_min)

            print('Convergence test failed, estimated minimum batch size:', self.Ns)
            return False

    def generate_data(self, Nd, Ns_cand):
        '''Generates additional data with NN warm start.'''
        print('')
        print('Generating data...')

        import warnings
        np.seterr(over='warn', divide='warn', invalid='warn')
        warnings.filterwarnings('error')

        N_states = self.problem.N_states

        X_OUT = np.empty((N_states,0))
        A_OUT = np.empty((N_states,0))
        V_OUT = np.empty((1,0))
        t_OUT = np.empty((1,0))

        Ns_sol = 0
        start_time = time.time()

        # ----------------------------------------------------------------------

        while X_OUT.shape[1] < Nd:
            # Picks random sample with largest gradient
            X0 = (self.ub - self.lb) * np.random.rand(N_states, Ns_cand) + self.lb
            max_idx = self.get_largest_A(np.zeros((1, Ns_cand)), X0, 1)
            X0 = X0[:, max_idx[-1]]

            bc = self.problem.make_bc(X0)

            # Integrates the closed-loop system (NN controller)

            SOL = solve_ivp(self.problem.dynamics, [0., self.t1], X0,
                            method=self.config.ODE_solver,
                            args=(self.eval_U,),
                            rtol=1e-04)

            V_guess, A_guess = self.bvp_guess(SOL.t.reshape(1,-1), SOL.y)

            try:
                # Solves the two-point boundary value problem

                X_aug_guess = np.vstack((SOL.y, A_guess, V_guess))
                SOL = solve_bvp(self.problem.aug_dynamics, bc, SOL.t, X_aug_guess,
                                verbose=1,
                                tol=self.config.data_tol,
                                max_nodes=self.config.max_nodes)
                if not SOL.success:
                    warnings.warn(Warning())

                Ns_sol += 1
                V = SOL.y[-1:] + self.problem.terminal_cost(SOL.y[:N_states,-1])

                t_OUT = np.hstack((t_OUT, SOL.x.reshape(1,-1)))
                X_OUT = np.hstack((X_OUT, SOL.y[:N_states]))
                A_OUT = np.hstack((A_OUT, SOL.y[N_states:2*N_states]))
                V_OUT = np.hstack((V_OUT, V))

            except Warning:
                pass

        print('Generated', X_OUT.shape[1], 'data from', Ns_sol,
            'BVP solutions in %.1f' % (time.time() - start_time), 'sec')

        data = {'t': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT,
                'U': self.problem.U_star(np.vstack((X_OUT, A_OUT)))
            }
        data.update({
            'A_scaled': 2.*(data['A'] - self.A_lb)/(self.A_ub - self.A_lb) - 1.,
            'U_scaled': 2.*(data['U'] - self.U_lb)/(self.U_ub - self.U_lb) - 1.,
            'V_scaled': 2.*(data['V'] - self.V_min)/(self.V_max - self.V_min) - 1.
            })
        return data

class hjb_network_t0(hjb_network):
    def make_eval_graph(self, t, X):
        '''Builds the NN computational graph.'''

        # (N_states, ?) matrix of linearly rescaled input values
        V = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        # Hidden layers
        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            V = tf.tanh(tf.matmul(W, V) + b)
        # The last layer is linear -> it's outside the loop
        W = self.weights[-1]
        b = self.biases[-1]
        V = tf.matmul(W, V) + b

        V_descaled = (self.V_max - self.V_min)*(V + 1.)/2. + self.V_min

        return V, V_descaled

    def predict_V(self, t, X):
        '''Run a TensorFlow Session to predict the value function at arbitrary
        space-time coordinates.'''
        return self.sess.run(self.V_pred, {self.X_tf: X})

    def predict_A(self, t, X):
        '''Run a TensorFlow Session to predict the value gradient at arbitrary
        space-time coordinates.'''
        return self.sess.run(self.dVdX, {self.X_tf: X})

    def get_largest_A(self, t, X, N):
        '''Partially sorts space-time points by the predicted gradient norm.'''
        _, max_idx = self.sess.run(self.largest_dVdX,
                                   {self.k_largest: N,
                                    self.X_tf: X})
        return max_idx

    def eval_U(self, t, X):
        '''(Near-)optimal feedback control for arbitrary inputs (t,X).'''
        return self.sess.run(self.U, {self.X_tf: X}).astype(np.float64)

    def bvp_guess(self, t, X, eval_U=False):
        '''Predicts value, costate, and control with one session call.'''
        if eval_U:
            V, A, U = self.sess.run((self.V_pred, self.dVdX, self.U), {self.X_tf: X})
            return V, A, U.astype(np.float64)
        else:
            return self.sess.run((self.V_pred, self.dVdX), {self.X_tf: X})

    def train(self, train_data, val_data):
        '''Implements training with L-BFGS.'''

        train_data.update({
            'U': self.problem.U_star(np.vstack((train_data['X'], train_data['A'])))
            })
        train_data.update({
            'A_scaled': 2.*(train_data['A'] - self.A_lb)/(self.A_ub - self.A_lb) - 1.,
            'U_scaled': 2.*(train_data['U'] - self.U_lb)/(self.U_ub - self.U_lb) - 1.,
            'V_scaled': 2.*(train_data['V'] - self.V_min)/(self.V_max - self.V_min) - 1.
            })

        val_data = {self.X_tf: val_data.pop('X'), self.V_tf: val_data.pop('V'),
                    self.A_tf: val_data.pop('A')}
        val_data.update({
            self.U_tf: self.problem.U_star(
                np.vstack((val_data[self.X_tf], val_data[self.A_tf]))
                )
            })

        # ----------------------------------------------------------------------

        # Gets training options from configuration file
        self.Ns = self.config.batch_size
        if self.Ns is None:
            self.Ns = train_data['X'].shape[1]

        Ns_C = self.config.Ns_scale
        Ns_cand = self.config.Ns_cand
        Ns_max = self.config.Ns_max

        conv_tol = self.config.conv_tol

        max_rounds = self.config.max_rounds
        min_rounds = self.config.min_rounds

        weight_A = self.config.weight_A
        self.weight_A_tf = tf.placeholder(tf.float32, shape=())

        weight_U = self.config.weight_U
        self.weight_U_tf = tf.placeholder(tf.float32, shape=())

        self.loss = self.loss_V
        if weight_A[0] >= 10.0 * np.finfo(float).eps:
            self.loss = self.loss + self.weight_A_tf * self.loss_A
        if weight_U[0] >= 10.0 * np.finfo(float).eps:
            self.loss = self.loss + self.weight_U_tf * self.loss_U

        # ----------------------------------------------------------------------

        # Sets up optimizer stuff
        self.grads_list = [None]*3
        optimizer = None

        train_err = []
        train_grad_err = []
        train_ctrl_err = []

        val_err = []
        val_grad_err = []
        val_ctrl_err = []

        round_iters = []

        errors_to_track = [train_err, train_grad_err, train_ctrl_err]
        fetches = [[self.MAE, self.grad_MRL2, self.ctrl_MRL2]]

        # ----------------------------------------------------------------------

        for round in range(1,max_rounds+1):
            # Generates new data if needed
            if self.Ns > train_data['X'].shape[1]:
                new_data = self.generate_data(
                    self.Ns - train_data['X'].shape[1], Ns_cand)
                for key in new_data.keys():
                    train_data.update({
                        key: np.hstack((train_data[key], new_data[key]))
                        })

            self.Ns = np.minimum(self.Ns, Ns_max)

            print('Optimization round', round, ':')
            print('Batch size =', self.Ns,
                  ', gradient weight = %1.1e' % (weight_A[round-1]),
                  ', control weight = %1.1e' % (weight_U[round-1]))

            # ------------------------------------------------------------------

            idx = np.random.choice(
                train_data['X'].shape[1], self.Ns, replace=False)

            tf_dict = {self.X_tf: train_data['X'][:,idx],
                       self.A_tf: train_data['A'][:,idx],
                       self.U_tf: train_data['U'][:,idx],
                       self.V_tf: train_data['V'][:,idx],
                       self.A_scaled_tf: train_data['A_scaled'][:,idx],
                       self.U_scaled_tf: train_data['U_scaled'][:,idx],
                       self.V_scaled_tf: train_data['V_scaled'][:,idx],
                       self.weight_A_tf: weight_A[round-1],
                       self.weight_U_tf: weight_U[round-1]}

            # Minimizes with L-BFGS
            BFGS_opts = {}
            for key in self.config.BFGS_opts.keys():
                BFGS_opts[key] = self.config.BFGS_opts[key][round-1]

            optimizer = self._train_L_BFGS_B(tf_dict,
                optimizer=optimizer,
                errors_to_track=errors_to_track,
                fetches=fetches,
                options=BFGS_opts)

            # ------------------------------------------------------------------

            loss_V, loss_A, loss_U = self.sess.run(
                (self.loss_V, self.loss_A, self.loss_U), tf_dict)
            print('')
            print('loss_V = %1.1e' % (loss_V),
                  ', loss_A = %1.1e' % (loss_A),
                  ', loss_U = %1.1e' % (loss_U))

            # If didn't track training errors, compute them now
            for error,fetch in zip((train_err,train_grad_err,train_ctrl_err),
                               (self.MRAE, self.grad_MRL2, self.ctrl_MRL2)):
                if error not in errors_to_track:
                    error.append(self.sess.run(fetch, tf_dict))

            round_iters.append(len(train_err))

            val_errs = self.sess.run(
                (self.MAE, self.grad_MRL2, self.ctrl_MRL2), val_data)

            val_err.append(val_errs[0])
            val_grad_err.append(val_errs[1])
            val_ctrl_err.append(val_errs[2])

            print('')
            print('Training MAE error = %1.1e' % (train_err[-1]))
            print('Validation MAE error = %1.1e' % (val_err[-1]))
            print('Training grad. MRL2 error = %1.1e' % (train_grad_err[-1]))
            print('Validation grad. MRL2 error = %1.1e' % (val_grad_err[-1]))
            print('Training ctrl. MRL2 error = %1.1e' % (train_ctrl_err[-1]))
            print('Validation ctrl. MRL2 error = %1.1e' % (val_ctrl_err[-1]))

            # ------------------------------------------------------------------

            if max_rounds > 1:
                if self.convergence_test(tf_dict,
                                         optimizer.grad_eval,
                                         epsilon=conv_tol,
                                         Ns_sub=int(self.Ns/8),
                                         C=Ns_C):
                    if round >= min_rounds:
                        print('Convergence test satisfied, stopping optimization.')
                        break
                    else:
                        print('Convergence test satisfied, but have not trained for minimum number of rounds.')
                        self.Ns *= Ns_C

        errors = (train_err, train_grad_err, train_ctrl_err,
                  val_err, val_grad_err, val_ctrl_err)
        return round_iters, errors

    def convergence_test(self, tf_dict, sample_grad, epsilon, Ns_sub=1024, C=2):
        '''Convergence test as described in the paper.'''
        print('')
        print('Running convergence test...')

        sample_grad = np.linalg.norm(sample_grad, ord=1)

        # Calculates sample variance for (a subsample of) the batch
        idx = np.random.choice(self.Ns, Ns_sub, replace=False)

        tf_dict.update({
            self.X_tf: tf_dict[self.X_tf][:,idx],
            self.A_scaled_tf: tf_dict[self.A_scaled_tf][:,idx],
            self.U_scaled_tf: tf_dict[self.U_scaled_tf][:,idx],
            self.V_scaled_tf: tf_dict[self.V_scaled_tf][:,idx],
            })

        sample_var = np.empty((Ns_sub, self.packed_loss_grad.shape[0]))
        for i in range(Ns_sub):
            sample_var[i] = self.sess.run(self.packed_loss_grad, {
                self.X_tf: tf_dict[self.X_tf][:,i:i+1],
                self.A_scaled_tf: tf_dict[self.A_scaled_tf][:,i:i+1],
                self.U_scaled_tf: tf_dict[self.U_scaled_tf][:,i:i+1],
                self.V_scaled_tf: tf_dict[self.V_scaled_tf][:,i:i+1],
                self.weight_A_tf: tf_dict[self.weight_A_tf],
                self.weight_U_tf: tf_dict[self.weight_U_tf]
                })

        sample_var = np.var(sample_var, axis=0, ddof=1, dtype=np.float64)
        sample_var = sample_var.sum()

        print('sample variance = %1.1e' % (sample_var))
        print('sample gradient = %1.1e' % (sample_grad))
        print('epsilon = %1.1e' % (epsilon), ', Ns_sub =', Ns_sub)

        if sample_var <= epsilon * Ns_sub * sample_grad:
            # Convergence condition satisfied
            return True
        else:
            Ns_min = int(sample_var / (epsilon * sample_grad))
            if self.Ns >= Ns_min:
                Ns_min = int(self.Ns * (Ns_min / Ns_sub))
            self.Ns = np.minimum(C*self.Ns, Ns_min)

            print('Convergence test failed, estimated minimum batch size:', self.Ns)
            return False

    def generate_data(self, Nd, Ns_cand):
        '''Generates additional data with NN warm start.'''
        print('')
        print('Generating data...')

        import warnings
        np.seterr(over='warn', divide='warn', invalid='warn')
        warnings.filterwarnings('error')

        N_states = self.problem.N_states

        X_OUT = np.empty((N_states,0))
        A_OUT = np.empty((N_states,0))
        V_OUT = np.empty((1,0))

        Ns_sol = 0
        start_time = time.time()

        while X_OUT.shape[1] < Nd:
            # Picks random sample with largest gradient
            X0 = (self.ub - self.lb) * np.random.rand(N_states, Ns_cand) + self.lb
            max_idx = self.get_largest_A(np.zeros((1, Ns_cand)), X0, 1)
            X0 = X0[:, max_idx[-1]]

            bc = self.problem.make_bc(X0)

            # Integrates the closed-loop system (NN controller)

            SOL = solve_ivp(self.problem.dynamics, [0., self.t1], X0,
                            method=self.config.ODE_solver,
                            args=(self.eval_U,),
                            rtol=1e-04)

            V_guess, A_guess = self.bvp_guess(SOL.t.reshape(1,-1), SOL.y)

            try:
                # Solves the two-point boundary value problem

                X_aug_guess = np.vstack((SOL.y, A_guess, V_guess))
                SOL = solve_bvp(self.problem.aug_dynamics, bc, SOL.t, X_aug_guess,
                                verbose=1,
                                tol=self.config.data_tol,
                                max_nodes=self.config.max_nodes)
                if not SOL.success:
                    warnings.warn(Warning())

                Ns_sol += 1
                V = SOL.y[-1:] + self.problem.terminal_cost(SOL.y[:N_states,-1])

                X_OUT = np.hstack((X_OUT, SOL.y[:N_states,0:1]))
                A_OUT = np.hstack((A_OUT, SOL.y[N_states:2*N_states,0:1]))
                V_OUT = np.hstack((V_OUT, V[:,0:1]))

            except Warning:
                pass

        print('Generated', X_OUT.shape[1], 'data from', Ns_sol,
            'BVP solutions in %.1f' % (time.time() - start_time), 'sec')

        data = {'X': X_OUT, 'A': A_OUT, 'V': V_OUT,
                'U': self.problem.U_star(np.vstack((X_OUT, A_OUT)))
            }
        data.update({
            'A_scaled': 2.*(data['A'] - self.A_lb)/(self.A_ub - self.A_lb) - 1.,
            'U_scaled': 2.*(data['U'] - self.U_lb)/(self.U_ub - self.U_lb) - 1.,
            'V_scaled': 2.*(data['V'] - self.V_min)/(self.V_max - self.V_min) - 1.
            })
        return data
