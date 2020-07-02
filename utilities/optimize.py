'''
Contains tools to interface TensorFlow with Scipy's L-BFGS.
Modifies the ExternalOptimizerInterace from

https://github.com/tensorflow/tensorflow/blob/r1.11/
    tensorflow/contrib/opt/python/training/external_optimizer.py

to allow reusing of pre-compiled gradient graphs and easy access
to evaluations of the minimized loss function and gradient.
'''

from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
import numpy as np

class ScipyOptimizerInterface(ExternalOptimizerInterface):
    _DEFAULT_METHOD = 'L-BFGS-B'

    def __init__(self,
                 loss,
                 grads_list=[None]*3,
                 var_list=None,
                 equalities=None,
                 inequalities=None,
                 var_to_bounds=None,
                 **optimizer_kwargs):
        """Initialize a new interface instance.
        Args:
          loss: A scalar `Tensor` to be minimized.
          var_list: Optional `list` of `Variable` objects to update to minimize
            `loss`.  Defaults to the list of variables collected in the graph
            under the key `GraphKeys.TRAINABLE_VARIABLES`.
          grads_list: Optional `list` of of TensorFlow gradient tensors of the loss
            with respect to the optimization `Variable`s. The list elements must be
              [loss_grads, equalities_grads, inequalities_grads]
          equalities: Optional `list` of equality constraint scalar `Tensor`s to be
            held equal to zero.
          inequalities: Optional `list` of inequality constraint scalar `Tensor`s
            to be held nonnegative.
          var_to_bounds: Optional `dict` where each key is an optimization
            `Variable` and each corresponding value is a length-2 tuple of
            `(low, high)` bounds. Although enforcing this kind of simple constraint
            could be accomplished with the `inequalities` arg, not all optimization
            algorithms support general inequality constraints, e.g. L-BFGS-B. Both
            `low` and `high` can either be numbers or anything convertible to a
            NumPy array that can be broadcast to the shape of `var` (using
            `np.broadcast_to`). To indicate that there is no bound, use `None` (or
            `+/- np.infty`). For example, if `var` is a 2x3 matrix, then any of
            the following corresponding `bounds` could be supplied:
            * `(0, np.infty)`: Each element of `var` held positive.
            * `(-np.infty, [1, 2])`: First column less than 1, second column less
              than 2.
            * `(-np.infty, [[1], [2], [3]])`: First row less than 1, second row less
              than 2, etc.
            * `(-np.infty, [[1, 2, 3], [4, 5, 6]])`: Entry `var[0, 0]` less than 1,
              `var[0, 1]` less than 2, etc.
          **optimizer_kwargs: Other subclass-specific keyword arguments.
        """

        self._loss = loss
        self._equalities = equalities or []
        self._inequalities = inequalities or []

        if var_list is None:
            self._vars = variables.trainable_variables()
        else:
            self._vars = list(var_list)

        packed_bounds = None
        if var_to_bounds is not None:
            left_packed_bounds = []
            right_packed_bounds = []
            for var in self._vars:
                shape = var.get_shape().as_list()
                bounds = (-np.infty, np.infty)
                if var in var_to_bounds:
                    bounds = var_to_bounds[var]
                left_packed_bounds.extend(list(np.broadcast_to(bounds[0], shape).flat))
                right_packed_bounds.extend(list(np.broadcast_to(bounds[1], shape).flat))
            packed_bounds = list(zip(left_packed_bounds, right_packed_bounds))
        self._packed_bounds = packed_bounds

        self._update_placeholders = [
            array_ops.placeholder(var.dtype) for var in self._vars
        ]
        self._var_updates = [
            var.assign(array_ops.reshape(placeholder, _get_shape_tuple(var)))
            for var, placeholder in zip(self._vars, self._update_placeholders)
        ]

        loss_grads, equalities_grads, inequalities_grads = grads_list
        if loss_grads is None:
            loss_grads = _compute_gradients(loss, self._vars)
        if equalities_grads is None:
            equalities_grads = [
                _compute_gradients(equality, self._vars)
                for equality in self._equalities
            ]
        if inequalities_grads is None:
            inequalities_grads = [
                _compute_gradients(inequality, self._vars)
                for inequality in self._inequalities
            ]

        self.optimizer_kwargs = optimizer_kwargs

        self._packed_var = self._pack(self._vars)
        self._packed_loss_grad = self._pack(loss_grads)
        self._packed_equality_grads = [
            self._pack(equality_grads) for equality_grads in equalities_grads
        ]
        self._packed_inequality_grads = [
            self._pack(inequality_grads) for inequality_grads in inequalities_grads
        ]

        dims = [_prod(_get_shape_tuple(var)) for var in self._vars]
        accumulated_dims = list(_accumulate(dims))
        self._packing_slices = [
            slice(start, end)
            for start, end in zip(accumulated_dims[:-1], accumulated_dims[1:])
        ]

        self._grads_list = [loss_grads, equalities_grads, inequalities_grads]

    def _minimize(self,
                  initial_val,
                  loss_grad_func,
                  equality_funcs,
                  equality_grad_funcs,
                  inequality_funcs,
                  inequality_grad_funcs,
                  packed_bounds,
                  step_callback,
                  optimizer_kwargs):

        def loss_grad_func_wrapper(x):
            loss, gradient = loss_grad_func(x)
            return loss, gradient.astype('float64')

        optimizer_kwargs = dict(optimizer_kwargs.items())
        method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

        constraints = []
        for func, grad_func in zip(equality_funcs, equality_grad_funcs):
            constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
        for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
            constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

        minimize_args = [loss_grad_func_wrapper, initial_val]
        minimize_kwargs = {
            'jac': True,
            'callback': step_callback,
            'method': method,
            'constraints': constraints,
            'bounds': packed_bounds,
        }

        for kwarg in minimize_kwargs:
            if kwarg in optimizer_kwargs:
                if kwarg == 'bounds':
                    # Special handling for 'bounds' kwarg since ability to specify bounds
                    # was added after this module was already publicly released.
                    raise ValueError(
                        'Bounds must be set using the var_to_bounds argument')
                raise ValueError(
                    'Optimizer keyword arg \'{}\' is set '
                    'automatically and cannot be injected manually'.format(kwarg))

        minimize_kwargs.update(optimizer_kwargs)

        from scipy.optimize import minimize
        result = minimize(*minimize_args, **minimize_kwargs)

        self.loss_eval = result.fun
        self.grad_eval = result.jac

        return result['x']


def _accumulate(list_):
    total = 0
    yield total
    for x in list_:
        total += x
        yield total


def _get_shape_tuple(tensor):
    return tuple(dim.value for dim in tensor.get_shape())


def _prod(array):
    prod = 1
    for value in array:
        prod *= value
    return prod


def _compute_gradients(tensor, var_list):
    grads = gradients.gradients(tensor, var_list)
    # tf.gradients sometimes returns `None` when it should return 0.
    return [
        grad if grad is not None else array_ops.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]
