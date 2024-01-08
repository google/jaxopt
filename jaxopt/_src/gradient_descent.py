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

"""Implementation of gradient descent in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Union

from dataclasses import dataclass

from jaxopt._src import base
from jaxopt._src.proximal_gradient import ProximalGradient, ProxGradState


@dataclass(eq=False)
class GradientDescent(ProximalGradient):
  """Gradient Descent solver.

  Attributes:
    fun: a smooth function of the form ``fun(parameters, *args, **kwargs)``,
      where ``parameters`` are the model parameters w.r.t. which we minimize
      the function and the rest are fixed auxiliary parameters.
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

    stepsize: a stepsize to use (if <= 0, use backtracking line search), or a
      callable specifying the **positive** stepsize to use at each iteration.
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.

    acceleration: whether to use acceleration (also known as FISTA) or not.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").
    """

  def init_state(self,
                 init_params: Any,
                 *args,
                 **kwargs) -> ProxGradState:
    """Initialize the solver state.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      state
    """
    return super().init_state(init_params, None, *args, **kwargs)

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of gradient descent.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    return super().update(params, state, None, *args, **kwargs)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()
    self.reference_signature = self.fun
