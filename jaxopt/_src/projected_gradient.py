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

"""Implementation of projected gradient descent in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

from dataclasses import dataclass

from jaxopt._src import base
from jaxopt._src import prox
from jaxopt._src.proximal_gradient import ProximalGradient, ProxGradState


@dataclass(eq=False)
class ProjectedGradient(base.IterativeSolver):
  """Projected gradient solver.

  This solver is a convenience wrapper around :class:`jaxopt.ProximalGradient`.

  Attributes:
    fun: a smooth function of the form ``fun(parameters, *args, **kwargs)``,
      where ``parameters`` are the model parameters w.r.t. which we minimize
      the function and the rest are fixed auxiliary parameters.
    projection: projection operator associated with the constraints.
      It should be of the form ``projection(params, hyperparams_proj)``.
      See ``jaxopt.projection`` for examples.
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

    stepsize: a stepsize to use (if <= 0, use backtracking line search),
      or a callable specifying the **positive** stepsize to use at each iteration.
    maxiter: maximum number of projected gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.

    acceleration: whether to use acceleration (also known as FISTA) or not.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
  """
  fun: Callable
  projection: Callable
  value_and_grad: bool = False
  has_aux: bool = False

  stepsize: Union[float, Callable] = 0.0
  maxiter: int = 500
  maxls: int = 15
  tol: float = 1e-3

  acceleration: bool = True
  decrease_factor: float = 0.5
  verbose: Union[bool, int] = False

  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None

  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params: Any,
                 hyperparams_proj: Optional[Any] = None,
                 *args,
                 **kwargs) -> ProxGradState:
    """Initialize the parameters and state.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams_proj: pytree containing hyperparameters of projection.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    return self._pg.init_state(init_params, hyperparams_proj, *args, **kwargs)

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams_proj: Optional[Any] = None,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of projected gradient.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams_proj: pytree containing hyperparameters of projection.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    return self._pg.update(params, state, hyperparams_proj, *args, **kwargs)

  def run(self,
          init_params: Any,
          hyperparams_proj: Optional[Any] = None,
          *args,
          **kwargs) -> base.OptStep:
    return self._pg.run(init_params, hyperparams_proj, *args, **kwargs)

  def optimality_fun(self, sol, hyperparams_proj, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._pg.optimality_fun(sol, hyperparams_proj, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()

    prox_fun = prox.make_prox_from_projection(self.projection)

    self._pg = ProximalGradient(fun=self.fun,
                                prox=prox_fun,
                                value_and_grad=self.value_and_grad,
                                has_aux=self.has_aux,
                                stepsize=self.stepsize,
                                maxiter=self.maxiter,
                                maxls=self.maxls,
                                tol=self.tol,
                                acceleration=self.acceleration,
                                decrease_factor=self.decrease_factor,
                                verbose=self.verbose,
                                implicit_diff=self.implicit_diff,
                                jit=self.jit,
                                unroll=self.unroll)
