# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGD solver with Armijo line search."""

from dataclasses import dataclass
from functools import partial

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

import jax
import jax.lax as lax
import jax.numpy as jnp

from jaxopt.tree_util import tree_add_scalar_mul, tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul, tree_zeros_like
from jaxopt.tree_util import tree_add, tree_sub
from jaxopt._src.tree_util import tree_single_dtype

from jaxopt._src import base
from jaxopt._src import loop


def wolfe_cond_violated(stepsize, coef, f_cur, f_next, grad_sqnorm):
  eps = jnp.finfo(f_next.dtype).eps
  return stepsize * coef * grad_sqnorm > f_cur - f_next + eps


def curvature_cond_violated(stepsize, coef, f_cur, f_next, grad_sqnorm):
  return f_next < f_cur - stepsize * (1.-coef) * grad_sqnorm


def armijo_line_search(fun_with_aux, unroll, jit,
                       goldstein, maxls,
                       params, f_cur, stepsize, grad,
                       coef, decrease_factor, increase_factor, max_stepsize,
                       args, kwargs):
  """Perform Armijo Line search from starting parameters, stepsize and gradient.

  Args:
    fun_with_aux: function to minimize.
    jit: whether to JIT-compile the line search loop.
    unroll: whether to unroll the line search loop (default: "auto").
    goldstein: boolean, whether to use Goldstein or not.
    maxls: maximum number of steps.
    params: current params to optimize.
    f_cur: value of the loss at ``params``.
    stepsize: initial guess for stepsize.
    grad: gradient at ``params``.
    coef: ``1-agressiveness``.
    decrease_factor: factor to increase stepsize.
    increase_factor: factor to decrease stepsize.
    max_stepsize: upper bound on stepsize.
    args,kwargs: additionals parameters passed to ``fun_with_aux``.

  Returns:
    stepsize: stepsize Armijo line search conditions
    next_params: params after gradient step
    f_next: loss after gradient step
  """
  # FIXME: (zramzi) this should return the number of iterations,
  # the number of function and gradient calls to have
  # these values available down the line.
  next_params = tree_add_scalar_mul(params, -stepsize, grad)
  f_next, _ = fun_with_aux(next_params, *args, **kwargs)
  grad_sqnorm = tree_l2_norm(grad, squared=True)

  def update_stepsize(t):
    """Multiply stepsize per factor, return new params and new value."""
    stepsize, factor = t
    stepsize = stepsize * factor
    stepsize = jnp.minimum(stepsize, max_stepsize)
    next_params = tree_add_scalar_mul(params, -stepsize, grad)
    f_next, _ = fun_with_aux(next_params, *args, **kwargs)
    return stepsize, next_params, f_next

  def body_fun(t):
    stepsize, next_params, f_next, _ = t

    violated = wolfe_cond_violated(stepsize, coef, f_cur, f_next, grad_sqnorm)
    stepsize, next_params, f_next = lax.cond(
      violated, update_stepsize,
      lambda _: (stepsize, next_params, f_next),
      operand=(stepsize, decrease_factor))

    if goldstein:
      goldstein_violated = curvature_cond_violated(stepsize, coef, f_cur,
                                                   f_next, grad_sqnorm)
      stepsize, next_params, f_next = lax.cond(
        goldstein_violated, update_stepsize,
        lambda _: (stepsize, next_params, f_next),
        operand=(stepsize, increase_factor))
      violated = jnp.logical_or(violated, goldstein_violated)

    return stepsize, next_params, f_next, violated

  init_val = stepsize, next_params, f_next, jnp.array(True)
  ret = loop.while_loop(cond_fun=lambda t: t[-1],  # check boolean violated
                        body_fun=body_fun,
                        init_val=init_val, maxiter=maxls,
                        unroll=unroll, jit=jit)
  return ret[:-1]   # remove boolean


class ArmijoState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    error: residuals of current estimate
    value: current value of the loss
    stepsize: current stepsize
    velocity: momentum term
  """
  iter_num: int
  error: float
  value: float
  aux: Optional[Any]
  stepsize: float
  velocity: Optional[Any]


@dataclass(eq=False)
class ArmijoSGD(base.StochasticSolver):
  """SGD with Armijo line search.

  This implementation assumes that the "interpolation property" holds, see for example Vaswani et al. 2019 (https://arxiv.org/abs/1905.09997):
    the global optimum over D must also be a global optimum for any finite sample of D
  This is typically achieved by overparametrized models (e.g neural networks)
  in classification tasks with separable classes, or on regression tasks without noise.
  In practice this algorithm works well outside this setting.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    value_and_grad: whether ``fun`` just returns the value (False) or both
      the value and gradient (True).
    has_aux: whether ``fun`` outputs auxiliary data or not.
      If ``has_aux`` is False, ``fun`` is expected to be
        scalar-valued.
      If ``has_aux`` is True, then we have one of the following
        two cases.
      If ``value_and_grad`` is False, the output should be
      ``value, aux = fun(...)``.
      If ``value_and_grad == True``, the output should be
      ``(value, aux), grad = fun(...)``.
      At each iteration of the algorithm, the auxiliary outputs are stored
        in ``state.aux``.

    aggressiveness: controls "agressiveness" of optimizer. (default: 0.9)
      Bigger values encourage bigger stepsize. Must belong to open interval
      (0,1).  If ``aggressiveness>0.5`` the learning_rate is guaranteed to be at
      least as big as ``min(1/L, max_stepsize)`` where ``L`` is the Lipschitz
      constant of the loss on the current batch.
    decrease_factor: factor by which to decrease the stepsize during line search
      (default: 0.8).
    increase_factor: factor by which to increase the stepsize during line search
      (default: 1.5).
    reset_option: strategy to use for resetting the stepsize at each iteration
      (default: "increase").

      - "conservative": re-use previous stepsize, producing a non increasing
        sequence of stepsizes. Slow convergence.
      - "increase": attempt to re-use previous stepsize multiplied by
        increase_factor. Cheap and efficient heuristic.
      - "goldstein": re-use previous stepsize and increase until curvature
        condition is fulfilled.  Higher runtime cost than "increase" but better
        theoretical guarantees.

    momentum: momentum parameter, 0 corresponding to no momentum.
    max_stepsize: a maximum step size to use. (default: 1.)
    pre_update: a function to execute before the solver's update.
      The function signature must be
      ``params, state = pre_update(params, state, *args, **kwargs)``.

    maxiter: maximum number of solver iterations.
    maxls: maximum number of steps in line search.
    tol: tolerance to use.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").

  References:
    Vaswani, S., Mishkin, A., Laradji, I., Schmidt, M., Gidel, G. and
    Lacoste-Julien, S., 2019.
    Painless stochastic gradient: Interpolation, line-search, and convergence
    rates.
    Advances in Neural Information Processing Systems 32.
  """
  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  aggressiveness: float = 0.9  # default value recommended by Vaswani et al.
  decrease_factor: float = 0.8  # default value recommended by Vaswani et al.
  increase_factor: float = 1.5  # default value recommended by Vaswani et al.
  reset_option: str = "increase"

  momentum: float = 0.0
  max_stepsize: float = 1.0

  pre_update: Optional[Callable] = None

  maxiter: int = 500
  maxls: int = 15
  tol: float = 1e-3
  verbose: Union[bool, int] = False

  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self, init_params, *args, **kwargs) -> ArmijoState:
    """Initialize the state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    if self.momentum == 0:
      velocity = None
    else:
      velocity = tree_zeros_like(init_params)

    value, aux = self._fun_with_aux(init_params, *args, **kwargs)

    params_dtype = tree_single_dtype(init_params)

    return ArmijoState(iter_num=jnp.asarray(0),
                       # Error should be dtype-compatible with the parameters,
                       # not with the value, since the error is derived from the
                       # gradient, which lives in the same space as the params.
                       error=jnp.asarray(jnp.inf, dtype=params_dtype),
                       value=jnp.asarray(jnp.inf, value.dtype),
                       aux=aux,
                       stepsize=jnp.asarray(self.max_stepsize,
                                            dtype=params_dtype),
                       velocity=velocity)

  def reset_stepsize(self, stepsize):
    """Return new step size for current step, according to reset_option."""
    if self.reset_option == 'goldstein':
      return stepsize
    if self.reset_option == 'conservative':
      return stepsize
    stepsize = stepsize * self.increase_factor
    return jnp.minimum(stepsize, self.max_stepsize)

  def update(self, params, state, *args, **kwargs) -> base.OptStep:
    """Performs one iteration of the solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    dtype = tree_single_dtype(params)

    if self.pre_update:
      params, state = self.pre_update(params, state, *args, **kwargs)

    stepsize = self.reset_stepsize(state.stepsize)

    (f_cur, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)

    goldstein = self.reset_option == 'goldstein'
    stepsize, next_params, f_next = self._armijo_line_search(
      goldstein, self.maxls, params, f_cur, stepsize, grad, self._coef,
      self.decrease_factor, self.increase_factor, self.max_stepsize, args,
      kwargs)

    if self.momentum == 0:
      next_velocity = None
    else:
      # next_params = params - stepsize*grad + momentum*(params - previous_params)
      velocity = tree_scalar_mul(self.momentum, state.velocity)
      next_params = tree_add(next_params, velocity)
      next_velocity = tree_sub(next_params, params)

    # error of last step, avoid recomputing a gradient
    error = tree_l2_norm(grad, squared=False)

    next_state = ArmijoState(iter_num=state.iter_num+1,
                             error=jnp.asarray(error, dtype=dtype),
                             value=jnp.asarray(f_next),
                             aux=aux,
                             stepsize=jnp.asarray(stepsize, dtype=dtype),
                             velocity=next_velocity)

    if self.verbose:
      self.log_info(
          next_state,
          error_name="Gradient Norm",
          additional_info={
              'Objective Value': next_state.value,
              'Stepsize': stepsize
          },
      )
    return base.OptStep(next_params, next_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)

  def _value_and_grad_fun(self, params, *args, **kwargs):
    (value, aux), grad = self._value_and_grad_with_aux(params, *args, **kwargs)
    return value, grad

  def _grad_fun(self, params, *args, **kwargs):
    return self._value_and_grad_fun(params, *args, **kwargs)[1]

  def __post_init__(self):
    super().__post_init__()

    options = ['increase', 'goldstein', 'conservative']
    if self.reset_option not in options:
      raise ValueError(f"'reset_option' should be one of {options}")
    if self.aggressiveness <= 0. or self.aggressiveness >= 1.:
      raise ValueError(f"'aggressiveness' must belong to open interval (0,1)")

    self._coef = 1 - self.aggressiveness

    self._fun_with_aux, _, self._value_and_grad_with_aux = \
      base._make_funs_with_aux(fun=self.fun,
                               value_and_grad=self.value_and_grad,
                               has_aux=self.has_aux)

    self.reference_signature = self.fun

    unroll = self._get_unroll_option()

    armijo_with_fun = partial(armijo_line_search, self._fun_with_aux, unroll,
                              self.jit)
    if self.jit:
      jitted_armijo = jax.jit(armijo_with_fun, static_argnums=(0,1))
      self._armijo_line_search = jitted_armijo
    else:
      self._armijo_line_search = armijo_with_fun
