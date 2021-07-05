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
from typing import NamedTuple
from typing import Optional

from dataclasses import dataclass

from jaxopt import base
from jaxopt import proximal_gradient2 as proximal_gradient


@dataclass
class GradientDescent(proximal_gradient.ProximalGradient):
  """Gradient Descent solver.

  Attributes:
    fun: a smooth function of the form ``fun(parameters, hyperparams, data)``,
      where ``parameters`` are the model parameters w.r.t. which we minimize
      the function and ``hyperparams`` are fixed auxiliary parameters.
    stepsize: a stepsize to use (if <= 0, use backtracking line search).
    maxiter: maximum number of proximal gradient descent iterations.
    maxls: maximum number of iterations to use in the line search.
    tol: tolerance to use.
    acceleration: whether to use acceleration (also known as FISTA) or not.
    verbose: whether to print error on every iteration or not.
      Warning: verbose=True will automatically disable jit.
    implicit_diff: if True, enable implicit differentiation using cg,
      if Callable, do implicit differentiation using callable as linear solver,
      if False, use autodiff through the solver implementation (note:
        this will unroll syntactic loops).
    has_aux: whether function fun outputs one (False) or more values (True).
      When True it will be assumed by default that fun(...)[0] is the objective.
  """

  def update(self,
             params: Any,
             state: NamedTuple,
             hyperparams: Optional[Any] = None,
             data: Optional[Any] = None) -> base.OptStep:
    """Performs one iteration of proximal gradient.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      hyperparams: pytree containing hyper-parameters, i.e.,
        differentiable arguments to be passed to ``fun``.
      data: pytree containing data, i.e.,
        non-differentiable arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    return super().update(params, state, (hyperparams, None), data)

  # pylint: disable=useless-super-delegation
  def run(self,
          init_params: Any,
          hyperparams: Optional[Any] = None,
          data: Optional[Any] = None) -> base.OptStep:
    """Runs gradient descent until convergence or max number of iterations.

    Args:
      init_params: pytree containing the initial parameters.
      hyperparams: pytree containing hyper-parameters, i.e.,
        differentiable arguments to be passed to ``fun``.
      data: pytree containing data, i.e.,
        non-differentiable arguments to be passed to ``fun``.
    Return type:
      base.OptStep
    Returns:
      (params, state)
    """
    return super().run(init_params, hyperparams, data)

  def optimality_fun(self, sol, hyperparams, data):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(sol, hyperparams, data)
