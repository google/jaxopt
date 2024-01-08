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

import dataclasses
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

import jax.numpy as jnp

from jaxopt._src import base
from jaxopt.tree_util import tree_add
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_zeros_like
from jaxopt._src.tree_util import tree_single_dtype


class PolyakSGDState(NamedTuple):
  """Named tuple containing state information."""
  iter_num: int
  error: float
  value: float
  aux: Any
  stepsize: float
  velocity: Optional[Any]

  num_fun_eval: int = 0
  num_grad_eval: int = 0


@dataclasses.dataclass(eq=False)
class PolyakSGD(base.StochasticSolver):
  r"""SGD with Polyak step size.

  The stochastic Polyak step-size is a simple and efficient step-size for SGD.
  Although this algorithm does not require to set a step-size parameter, it does
  require knowledge of a lower bound on the objective function (see below).
  Furthermore, some variants accept other hyperparameters. 

  .. warning::
      This method requires knowledge of an approximate value of the of the objective function
      minimum, passed through the ``fun_min`` argument. For overparametrized models, this can be
      set to 0 (default value). Failing to set an appropriate value for ``fun_min`` can lead to
      a model that diverges or converges to a suboptimal solution.


  This class implements two different variants of the stochastic Polyak step size method: ``SPS_max``
  and ``SPS+``. The ``SPS_max`` variant from (Loizou et al. 2021) accepts the hyperparameters
  ``max_stepsize`` and ``delta`` and sets the current step-size :math:`\gamma`  as

  .. math::

    \gamma = \min\left\{\frac{\text{fun}(x) - \text{fun}(x^\star)}{\|\nabla \text{fun}(x)\|^2 + \text{delta}}, \text{max_stepsize}\right\}

  while for the ``SPS+`` variant, it is given by

  .. math::

    \gamma = \max\left\{0, \frac{\text{fun}(x) - \text{fun}(x^\star)}{\|\nabla \text{fun}(x)\|^2}\right\}

  and the step-size is zero whenever :math:`\|\nabla \text{fun}(x)\|^2` is zero.

  In all cases, the step size is then used in the update

  .. math::

    v_{t+1} &= \text{momentum} v_t - \gamma \nabla \text{fun}(x) \\
    x_{t+1} &= x_t + v_{t+1}


  Attributes:
    fun: a function of the form ``fun(params, *args, **kwargs)``, where
      ``params`` are parameters of the model,
      ``*args`` and ``**kwargs`` are additional arguments.
    value_and_grad: whether ``fun`` just returns the value (False) or both the
      value and gradient (True).
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
    fun_min: a lower bound on fun.

    variant: which version of the stochastic Polyak step-size is implemented.
      Can be one of "SPS_max" or "SPS+".
    max_stepsize: a maximum step size to use. Only used when variant="SPS_max".
    delta: a value to add in the denominator of the update (default: 0).
    momentum: momentum parameter, 0 corresponding to no momentum.
    pre_update: a function to execute before the solver's update.
      The function signature must be
      ``params, state = pre_update(params, state, *args, **kwargs)``.

    maxiter: maximum number of solver iterations.
    tol: tolerance to use.
    verbose: whether to print information on every iteration or not.

    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.

    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto").

  References:
    Berrada, Leonard and Zisserman, Andrew and Kumar, M Pawan.
    `"Training neural networks for and by interpolation" <https://arxiv.org/abs/1906.05661>`_.
    International Conference on Machine Learning, 2020.


    Loizou, Nicolas and Vaswani, Sharan and Laradji, Issam Hadj and
    Lacoste-Julien, Simon.
    `"Stochastic polyak step-size for sgd: An adaptive learning rate for fast
    convergence" <https://arxiv.org/abs/2002.10542>`_.
    International Conference on Artificial Intelligence and Statistics, 2021.
  """
  fun: Callable
  value_and_grad: bool = False
  has_aux: bool = False
  fun_min: float = 0.0

  variant: str = "SPS_max"
  max_stepsize: float = 1.0
  delta: float = 0.0
  momentum: float = 0.0
  pre_update: Optional[Callable] = None

  maxiter: int = 500
  tol: float = 1e-3
  verbose: Union[bool, int] = False

  implicit_diff: bool = False
  implicit_diff_solve: Optional[Callable] = None

  jit: bool = True
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
    value, aux = self._fun(init_params, *args, **kwargs)

    if self.momentum == 0:
      velocity = None
    else:
      velocity = tree_zeros_like(init_params)

    params_dtype = tree_single_dtype(init_params)

    return PolyakSGDState(iter_num=jnp.asarray(0),
                          error=jnp.asarray(jnp.inf, dtype=params_dtype),
                          value=jnp.asarray(jnp.inf, dtype=value.dtype),
                          stepsize=jnp.asarray(1.0, dtype=params_dtype),
                          aux=aux,
                          velocity=velocity,
                          num_fun_eval=jnp.array(1, base.NUM_EVAL_DTYPE),
                          num_grad_eval=jnp.array(0, base.NUM_EVAL_DTYPE))

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
    dtype = tree_single_dtype(params)

    if self.pre_update:
      params, state = self.pre_update(params, state, *args, **kwargs)

    (value, aux), grad = self._value_and_grad_fun(params, *args, **kwargs)

    grad_sqnorm = tree_l2_norm(grad, squared=True)
    if self.variant == "SPS_max":
      stepsize = jnp.minimum(
          (value - self.fun_min) / (grad_sqnorm + self.delta), self.max_stepsize
      )
    elif self.variant == "SPS+":
      # if grad_sqnorm is smaller than machine epsilon, we set the stepsize to 0
      stepsize = jnp.where(
          grad_sqnorm <= jnp.finfo(dtype).eps,
          0.0,
          jnp.maximum((value - self.fun_min) / grad_sqnorm, 0),
      )
    else:
      raise NotImplementedError(f"Unknown variant {self.variant}")

    stepsize = stepsize.astype(state.stepsize.dtype)

    if self.momentum == 0:
      new_params = tree_add_scalar_mul(params, -stepsize, grad)
      new_velocity = None
    else:
      # new_v = momentum * v - step_size * grad
      # new_params = params + new_v
      new_velocity = tree_sub(tree_scalar_mul(self.momentum, state.velocity),
                              tree_scalar_mul(stepsize, grad))
      new_params = tree_add(params, new_velocity)

    error = jnp.sqrt(grad_sqnorm)
    new_state = PolyakSGDState(iter_num=state.iter_num + 1,
                               error=jnp.asarray(error, dtype=dtype),
                               velocity=new_velocity,
                               value=jnp.asarray(value),
                               stepsize=jnp.asarray(stepsize, dtype=dtype),
                               aux=aux,
                               num_fun_eval=state.num_fun_eval + 1,
                               num_grad_eval=state.num_grad_eval + 1)

    if self.verbose:
      self.log_info(
          new_state,
          error_name="Gradient Norm",
          additional_info={
              "Objective Value": value,
              "Stepsize": stepsize,
          }
      )
    return base.OptStep(params=new_params, state=new_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    return self._grad_fun(params, *args, **kwargs)[0]

  def __hash__(self):
    # We assume that the attribute values completely determine the solver.
    return hash(self.attribute_values())

  def __post_init__(self):
    super().__post_init__()

    self._fun, self._grad_fun, self._value_and_grad_fun = (
        base._make_funs_with_aux(
            fun=self.fun,
            value_and_grad=self.value_and_grad,
            has_aux=self.has_aux,
        )
    )

    self.reference_signature = self.fun
