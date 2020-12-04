'''
Run this script to train the NN. Loads the problem and configuration
according to examples/choose_problem.py.
'''

import numpy as np
import tensorflow as tf
import scipy.io
import time

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from utilities.neural_networks import HJBnet
else:
    from utilities.neural_networks import HJBnet_t0 as HJBnet

np.random.seed(config.random_seeds['train'])
tf.set_random_seed(config.random_seeds['train'])

# ---------------------------------------------------------------------------- #

##### Loads data sets #####

train_data = scipy.io.loadmat('examples/' + system + '/data_train.mat')
test_data = scipy.io.loadmat('examples/' + system + '/data_test.mat')

if time_dependent:
    system += '/tspan'
else:
    system += '/t0'
    for data in [train_data, test_data]:
        idx0 = np.nonzero(np.equal(data.pop('t'), 0.))[1]
        data.update({'X': data['X'][:,idx0],
                     'A': data['A'][:,idx0],
                     'V': data['V'][:,idx0]})

for data in [train_data, test_data]:
    data['U'] = problem.U_star(np.vstack((data['X'], data['A'])))

N_train = train_data['X'].shape[1]
N_test = test_data['X'].shape[1]

print('')
print('Number of training data:', N_train)
print('Number of test data:', N_test)
print('')

# ---------------------------------------------------------------------------- #

##### Builds and trains the neural net #####

model_path = 'examples/' + system + '/V_model.mat'

if int_input('Load pre-trained model? Enter 0 for no, 1 for yes:'):
    # Loads pre-trained model
    parameters, scaling = load_NN(model_path)
else:
    # Initializes the model from scratch
    parameters = None
    scaling = {
        'lb': np.min(train_data['X'], axis=1, keepdims=True),
        'ub': np.max(train_data['X'], axis=1, keepdims=True),
        'A_lb': np.min(train_data['A'], axis=1, keepdims=True),
        'A_ub': np.max(train_data['A'], axis=1, keepdims=True),
        'U_lb': np.min(train_data['U'], axis=1, keepdims=True),
        'U_ub': np.max(train_data['U'], axis=1, keepdims=True),
        'V_min': np.min(train_data['V']),
        'V_max': np.max(train_data['V'])
        }

start_time = time.time()

model = HJBnet(problem, scaling, config, parameters)

iters, errors = model.train(train_data, test_data)

train_time = time.time() - start_time
print('Computation time: %.0f' % (train_time), 'sec')

# ---------------------------------------------------------------------------- #

save_dict = {'train_time': train_time, 'round_iters': iters,
             'train_err': errors[0],
             'train_grad_err': errors[1],
             'train_ctrl_err': errors[2],
             'test_err': errors[3],
             'test_grad_err': errors[4],
             'test_ctrl_err': errors[5]}
scipy.io.savemat('examples/' + system + '/results/train_results.mat', save_dict)

# Saves model parameters
save_me = int_input('Save model parameters? Enter 0 for no, 1 for yes:')

if save_me:
    weights, biases = model.export_model()
    save_dict = scaling
    save_dict.update({'weights': weights,
                      'biases': biases,
                      'train_time': train_time,
                      'test_grad_err': errors[4][-1],
                      'test_ctrl_err': errors[5][-1]})
    scipy.io.savemat(model_path, save_dict)
