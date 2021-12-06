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

"""SGD solver with Polyak step size."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import dataclasses

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_zeros_like


class PolyakSGDState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float
  value: float
  aux: Any
  stepsize: float
  velocity: Optional[Any]


@dataclasses.dataclass(eq=False)
class PolyakSGD(base.StochasticSolver):
  """SGD with Polyak step size.

  This solver computes step sizes in an adaptive manner. If the computed step
  size at a given iteration is smaller than ``max_stepsize``, it is accepted.
  Otherwise, ``max_stepsize`` is used. This ensures that the solver does not
  take over-confident steps. This is why ``max_stepsize`` is the most important
  hyper-parameter.

  This implementation assumes that the interpolation property holds.

  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    max_stepsize: a maximum step size to use.
    delta: a value to add in the denominator of the update (default: 0).
    momentum: momentum parameter, 0 corresponding to no momentum.
    pre_update: a function to execute before the solver's update.
      The function signature must be
      ``params, state = pre_update(params, state, *args, **kwargs)``.
    maxiter: maximum number of solver iterations.
    tol: tolerance to use.
    verbose: whether to print error on every iteration or not. verbose=True will
      automatically disable jit.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether ``fun`` outputs one (False) or more values (True).
      When True it will be assumed by default that ``fun(...)[0]``
      is the objective value. The auxiliary outputs are stored in
      ``state.aux``.
    jit: whether to JIT-compile the optimization loop (default: "auto").
    unroll: whether to unroll the optimization loop (default: "auto").

  References:
    Berrada, Leonard and Zisserman, Andrew and Kumar, M Pawan.
    "Training neural networks for and by interpolation".
    International Conference on Machine Learning, 2020.
    https://arxiv.org/abs/1906.05661

    Loizou, Nicolas and Vaswani, Sharan and Laradji, Issam Hadj and
    Lacoste-Julien, Simon.
    "Stochastic polyak step-size for sgd: An adaptive learning rate for fast
    convergence".
    International Conference on Artificial Intelligence and Statistics, 2021.
    https://arxiv.org/abs/2002.10542
  """
  fun: Callable
  max_stepsize: float = 1.0
  delta: float = 0.0
  momentum: float = 0.0
  pre_update: Optional[Callable] = None
  maxiter: int = 500
  tol: float = 1e-3
  verbose: int = 0
  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Any,
                 *args,
                 **kwargs) -> PolyakSGDState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    del args, kwargs  # Not used.

    if self.momentum == 0:
      velocity = None
    else:
      velocity = tree_zeros_like(init_params)

    return PolyakSGDState(iter_num=jnp.asarray(0),
                          error=jnp.asarray(jnp.inf),
                          value=jnp.asarray(jnp.inf),
                          stepsize=jnp.asarray(1.0),
                          aux=None,
                          velocity=velocity)

  def update(self,
             params: Any,
             state: PolyakSGDState,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the solver.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    if self.pre_update:
      params, state = self.pre_update(params, state, *args, **kwargs)

    (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)

    grad_sqnorm = tree_l2_norm(grad, squared=True)
    stepsize = jax.lax.min(value / (grad_sqnorm + self.delta),
                           self.max_stepsize)

    if self.momentum == 0:
      new_params = tree_add_scalar_mul(params, -stepsize, grad)
      new_velocity = None
    else:
      # new_v = momentum * v - step_size * grad
      # new_params = params + new_v
      new_velocity = tree_sub(tree_scalar_mul(self.momentum, state.velocity),
                              tree_scalar_mul(stepsize, grad))
      new_params = tree_add(params, new_velocity)

    new_state = PolyakSGDState(iter_num=state.iter_num + 1,
                               error=jnp.sqrt(grad_sqnorm),
                               velocity=new_velocity,
                               value=value,
                               stepsize=stepsize,
                               aux=aux)
    return base.OptStep(params=new_params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)[0]

  def __hash__(self):
    # We assume that the attribute values completely determine the solver.
    return hash(self.attribute_values())

  def __post_init__(self):
    # Handle has_aux once for all.
    if self.has_aux:
      fun_with_aux = self.fun
    else:
      fun_with_aux = lambda p, *a, **kw: (self.fun(p, *a, **kw), None)

    # Pre-compile useful functions.
    self._value_and_grad_fun = jax.value_and_grad(fun_with_aux, has_aux=True)
    self._grad_fun = jax.grad(fun_with_aux, has_aux=True)

    self.reference_signature = self.fun
