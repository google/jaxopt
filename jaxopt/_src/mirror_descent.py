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

"""Implementation of mirror descent in JAX."""

import inspect

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src.tree_util import tree_add_scalar_mul
from jaxopt._src.tree_util import tree_l2_norm
from jaxopt._src.tree_util import tree_sub


class MirrorDescentState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float
  aux: Optional[Any] = None

  num_fun_eval: int = 0
  num_grad_eval: int = 0
  num_proj_eval: int = 0


@dataclass(eq=False)
class MirrorDescent(base.IterativeSolver):
  """Mirror descent solver.

  This solver minimizes:
    argmin_x fun(x, *args, **kwargs),
  where fun is smooth with convex domain.

  The stopping criterion is:
    ||x - projection_grad(x, g, 1.0, hyperparams_proj)||_2 <= tol,
  where ``g = grad(fun)(x, *args, **kwargs)``.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)``.
    projection_grad: a function of the form
      ``projection_grad(x, g, stepsize, hyperparams_proj)`` representing the
      mirror descent update for iterate x and gradient g. Optionally, it can be
      instantiated from a projection and mapping function (mirror map) using the
      method `make_projection_grad`.
    stepsize: a stepsize to use, or a callable specifying the stepsize to use at
      each iteration.
    maxiter: maximum number of mirror descent iterations.
    tol: tolerance to use.
    verbose: whether to print information on every iteration or not.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").

  References:
    Nemirovskij, Arkadij Semenovič, and David Borisovich Yudin. "Problem
    complexity and method efficiency in optimization." J. Wiley @ Sons, New
    York(1983).
  """
  fun: Callable
  projection_grad: Optional[Callable]
  stepsize: Union[float, Callable]
  maxiter: int = 500
  tol: float = 1e-2
  verbose: Union[bool, int] = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  @staticmethod
  def make_projection_grad(projection: Callable,
                           mapping_fun: Callable) -> Callable:
    """Instantiates `projection_grad` argument from projection and mirror map.

    Args:
      projection: projection operator of the form
        ``projection(x, hyperparams_proj)``, typically
        ``argmin_z D_{gen_fun}(z, mapping_fun^{-1}(y))``.
      mapping_fun: the mirror map, typically of the form
        ``mapping_fun = grad(gen_fun)``, where `gen_fun` is the generating
        function of the Bregman divergence.
    Returns:
      A function `projection_grad(x, g, stepsize, hyperparams_proj)`
      representing the mirror descent update for iterate x and gradient g.
    """
    def projection_grad(x, x_fun_grad, stepsize, hyperparams_proj):
      update = tree_add_scalar_mul(mapping_fun(x), -stepsize, x_fun_grad)
      return projection(update, hyperparams_proj)
    return projection_grad

  def init_state(self,
                 init_params: Any,
                 hyperparams_proj: Any,
                 *args,
                 **kwargs) -> base.OptStep:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
    Returns:
      state
    """
    del hyperparams_proj  # Not used.

    if self.has_aux:
      _, aux = self.fun(init_params, *args, **kwargs)
      num_fun_eval = jnp.array(1, base.NUM_EVAL_DTYPE)
    else:
      aux = None
      num_fun_eval = jnp.array(0, base.NUM_EVAL_DTYPE)

    return MirrorDescentState(iter_num=jnp.asarray(0),
                              error=jnp.asarray(jnp.inf),
                              aux=aux,
                              num_fun_eval=num_fun_eval,
                              num_grad_eval=jnp.array(0, base.NUM_EVAL_DTYPE),
                              num_proj_eval=jnp.array(0, base.NUM_EVAL_DTYPE))

  def _error(self, x, next_x, stepsize):
    diff_x = tree_sub(next_x, x)
    diff_norm = tree_l2_norm(diff_x)
    return diff_norm / stepsize

  def _stepsize(self, iter_num):
    if isinstance(self.stepsize, Callable):
      return self.stepsize(iter_num)
    return self.stepsize

  def _update(self, x, state, hyperparams_proj, args, kwargs):
    iter_num = state.iter_num
    stepsize = self._stepsize(iter_num)
    x_fun_grad, aux = self._grad_with_aux(x, *args, **kwargs)
    next_x = self.projection_grad(x, x_fun_grad, stepsize, hyperparams_proj)
    error = self._error(x, next_x, stepsize)
    next_state = MirrorDescentState(
      iter_num=iter_num + 1,
      error=error,
      aux=aux,
      num_fun_eval=state.num_fun_eval + 1,
      num_grad_eval=state.num_grad_eval + 1,
      num_proj_eval=state.num_proj_eval + 1,)
    
    if self.verbose:
      self.log_info(
          next_state,
          error_name="Distance btw Iterates"
      )
    return base.OptStep(params=next_x, state=next_state)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams_proj: Any,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of mirror descent.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams_proj: pytree containing hyperparameters of projection.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    return self._update(params, state, hyperparams_proj, args, kwargs)

  def run(self,
          init_params: Any,
          hyperparams_proj: Optional[Any] = None,
          *args,
          **kwargs) -> base.OptStep:
    return super().run(init_params, hyperparams_proj, *args, **kwargs)

  def _fixed_point_fun(self, sol, hyperparams_proj, args, kwargs):
    sol_fun_grad, _ = self._grad_with_aux(sol, *args, **kwargs)
    return self.projection_grad(sol, sol_fun_grad, 1.0, hyperparams_proj)

  def optimality_fun(self, sol, hyperparams_proj, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    fp = self._fixed_point_fun(sol, hyperparams_proj, args, kwargs)
    return tree_sub(fp, sol)

  def __post_init__(self):
    super().__post_init__()

    if self.has_aux:
      fun_with_aux = self.fun
    else:
      fun_with_aux = lambda *a, **kw: (self.fun(*a, **kw), None)

    self._grad_with_aux = jax.grad(fun_with_aux, has_aux=True)

    # Sets up reference signature.
    fun = getattr(self.fun, "subfun", self.fun)
    signature = inspect.signature(fun)
    parameters = list(signature.parameters.values())
    new_param = inspect.Parameter(name="hyperparams_proj",
                                  kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    parameters.insert(1, new_param)
    self.reference_signature = inspect.Signature(parameters)
