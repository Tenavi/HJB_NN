import numpy as np
import tensorflow as tf
import scipy.io
import time

from utilities.other import int_input, load_NN
from examples.choose_problem import system, problem, config, time_dependent

if time_dependent:
    from utilities.neural_networks import hjb_network
else:
    from utilities.neural_networks import hjb_network_t0 as hjb_network

np.random.seed(config.random_seeds['train'])
tf.set_random_seed(config.random_seeds['train'])

# ---------------------------------------------------------------------------- #

train_data = scipy.io.loadmat('examples/' + system + '/data_train.mat')
val_data = scipy.io.loadmat('examples/' + system + '/data_val.mat')

if time_dependent:
    system += '/tspan'
else:
    system += '/t0'
    for data in [train_data, val_data]:
        idx0 = np.nonzero(np.equal(data.pop('t'), 0.))[1]
        data.update({'X': data['X'][:,idx0],
                     'A': data['A'][:,idx0],
                     'V': data['V'][:,idx0]})

N_train = train_data['X'].shape[1]
N_val = val_data['X'].shape[1]

print('')
print('Number of training data:', N_train)
print('Number of validation data:', N_val)
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
        'lb': train_data.pop('lb'), 'ub': train_data.pop('ub'),
        'A_lb': train_data.pop('A_lb'), 'A_ub': train_data.pop('A_ub'),
        'U_lb': train_data.pop('U_lb'), 'U_ub': train_data.pop('U_ub'),
        'V_min': train_data.pop('V_min'), 'V_max': train_data.pop('V_max')
        }

start_time = time.time()

model = hjb_network(problem, scaling, config, parameters)

iters, errors = model.train(train_data, val_data, config.training_opts)

train_time = time.time() - start_time
print('Computation time: %.0f' % (train_time), 'sec')

# ---------------------------------------------------------------------------- #

save_dict = {'train_time': train_time, 'round_iters': iters,
             'train_err': errors[0], 'train_grad_err': errors[1],
             'train_ctrl_err': errors[2],
             'val_err': errors[3], 'val_grad_err': errors[4],
             'val_ctrl_err': errors[5]}
scipy.io.savemat('examples/' + system + '/results/train_results.mat', save_dict)

# Saves model parameters
save_me = int_input('Save model parameters? Enter 0 for no, 1 for yes:')

if save_me:
    weights, biases = model.export_model()
    save_dict = scaling
    save_dict.update({'weights': weights,
                      'biases': biases,
                      'train_time': train_time,
                      'val_grad_err': errors[4],
                      'val_ctrl_err': errors[5]})
    scipy.io.savemat(model_path, save_dict)
