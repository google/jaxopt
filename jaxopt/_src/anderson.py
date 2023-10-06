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

"""Anderson Acceleration for finding a fixed point in JAX."""

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import List
from typing import Union

from typing import Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxopt._src import base
from jaxopt._src import linear_solve
from jaxopt._src.tree_util import tree_l2_norm, tree_sub
from jaxopt._src.tree_util import tree_vdot, tree_add
from jaxopt._src.tree_util import tree_mul, tree_scalar_mul
from jaxopt._src.tree_util import tree_average, tree_add_scalar_mul
from jaxopt._src.tree_util import tree_map, tree_gram


def minimize_residuals(residual_gram, ridge):
  """Return the solution to the linear system with slack variable."""
  m = residual_gram.shape[0]
  residual_gram = residual_gram + ridge * jnp.eye(m) # avoid ill-posed systems
  H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, m))],
                  [ jnp.ones((m, 1)), residual_gram  ]])
  c = jnp.zeros((m+1)).at[0].set(1)
  alphas = jnp.linalg.solve(H, c)
  return alphas


def anderson_step(params_history, residuals_history,
                  residual_gram, ridge, beta):
  """Produces new Anderson iterate from two sequences.

  Args:
    params_history: each colum  i must contain f_h_i=f(p_h_i)
    residual_gram: Gram matrix of residuals
  """
  alphas = minimize_residuals(residual_gram, ridge)
  alphas = alphas[1:]  # drop dummy variable (constraint satisfaction)
  pa = tree_average(params_history, alphas)
  ra = tree_average(residuals_history, alphas)
  extrapolated = tree_add_scalar_mul(pa, beta, ra)
  return extrapolated


def pytree_replace_elem(tree_batched, index, new_elems):
  """Replace an element of a batch stored in a pytree.

  Args:
    tree: tree of arrays of shape (batch_size, ...)
    index: an integer between ``0`` and ``batch_size-1``
    values: a pytree with the same structure as ``tree``
      but without the additional batch dimension on each leaf

  Returns:
    the same pytree with the element at pos ``index`` replaced
  """
  at_set = lambda x, elem: x.at[index].set(elem)
  return tree_map(at_set, tree_batched, new_elems)


def update_history(pos, params_history, residuals_history,
                   residual_gram, extrapolated, residual):
  params_history = pytree_replace_elem(params_history, pos, extrapolated)
  residuals_history = pytree_replace_elem(residuals_history, pos, residual)
  new_row = jax.vmap(tree_vdot, in_axes=(0, None))(residuals_history, residual)
  residual_gram = residual_gram.at[pos,:].set(new_row)
  residual_gram = residual_gram.at[:,pos].set(new_row)
  error = jnp.sqrt(residual_gram[pos,pos])
  return params_history, residuals_history, residual_gram, error


class AndersonState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number
    error: residuals of current estimate
    params_history: history of previous anderson iterates
    residuals_history: residuals of previous iterates
      fixed_point_fun(params_history) - params_history
    residual_gram: Gram matrix: G.T @ G with G the matrix of residuals
      each column of G is a flattened pytree of residuals_history
    aux: auxiliary data returned by fixed_point_fun
  """
  iter_num: int
  error: float
  params_history: Any
  residuals_history: Any
  residual_gram: jnp.ndarray
  aux: Optional[Any] = None

  num_fun_eval: int = 0


@dataclass(eq=False)
class AndersonAcceleration(base.IterativeSolver):
  """Anderson acceleration.

  Attributes:
    fixed_point_fun: a function ``fixed_point_fun(x, *args, **kwargs)``
      returning a pytree with the same structure and type as x
      See the reference below for conditions that the function must fulfill
      in order to guarantee convergence.
      In particular, if the Banach fixed point theorem
      conditions hold, Anderson acceleration will converge.
    history_size: size of history. Affect memory cost.
    mixing_frequency: frequency of Anderson updates. (default: 1).
      Only one every `mixing_frequency` updates uses Anderson, while the other
      updates use regular fixed point iterations.
    beta: momentum in Anderson updates. Default = 1.
    maxiter: maximum number of iterations.
    tol: tolerance (stoping criterion).
    ridge: ridge regularization in solver.
      Consider increasing this value if the solver returns ``NaN``.
    has_aux: wether fixed_point_fun returns additional data. (default: False)
      This additional data is not taken into account by the fixed point.
      The solver returns the `aux` associated to the last iterate (i.e the fixed point).
    verbose: whether to print information on every iteration or not.
    implicit_diff: whether to enable implicit diff or autodiff of unrolled
      iterations.
    implicit_diff_solve: the linear system solver to use.
    jit: whether to JIT-compile the optimization loop (default: True).
    unroll: whether to unroll the optimization loop (default: "auto")

  References:
    Pollock, Sara, and Leo Rebholz.
    "Anderson acceleration for contractive and noncontractive operators."
    arXiv preprint arXiv:1909.04638 (2019).
  """
  fixed_point_fun: Callable
  history_size: int = 5
  mixing_frequency: int = 1
  beta: float = 1.
  maxiter: int = 100
  tol: float = 1e-5
  ridge: float = 1e-5
  has_aux: bool = False
  verbose: Union[bool, int] = False
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: bool = True
  unroll: base.AutoOrBoolean = "auto"

  def init_state(self,
                 init_params,
                 *args,
                 **kwargs) -> AndersonState:
    """Initialize the solver state.

    Args:
      init_params: initial guess of the fixed point, pytree
      *args: additional positional arguments to be passed to ``fixed_point_fun``.
      **kwargs: additional keyword arguments to be passed to ``fixed_point_fun``.
    Returns:
      state
    """
    m = self.history_size
    params_history = tree_map(lambda x: jnp.tile(x, [m]+[1]*x.ndim),
                              init_params)
    residuals_history = tree_map(jnp.zeros_like, params_history)
    residual_gram = jnp.zeros((m,m))

    _, aux = self._value_with_aux(init_params, *args, **kwargs)

    return AndersonState(iter_num=jnp.asarray(0),
                         error=jnp.asarray(jnp.inf),
                         params_history=params_history,
                         residuals_history=residuals_history,
                         residual_gram=residual_gram,
                         aux=aux,
                         num_fun_eval=jnp.array(1, base.NUM_EVAL_DTYPE))

  def update(self,
             params: Any,
             state: NamedTuple,
             *args,
             **kwargs) -> base.OptStep:
    """Performs one iteration of the Anderson acceleration.

    Args:
      params: pytree containing the parameters.
      state: named tuple containing the solver state.
      *args: additional positional arguments to be passed to ``fixed_point_fun``.
      **kwargs: additional keyword arguments to be passed to ``fixed_point_fun``.
    Returns:
      (params, state)
    """

    iter_num = state.iter_num
    anderson_freq = jnp.equal(jnp.mod(iter_num, self.mixing_frequency), 0)
    is_not_init = jnp.greater_equal(iter_num, self.history_size)

    def perform_anderson_step(t):
      _, state = t
      extrapolated = anderson_step(state.params_history, state.residuals_history,
                                   state.residual_gram,
                                   self.ridge, self.beta)
      return extrapolated

    def use_param(t):
      return t[0]

    extrapolated = jax.lax.cond(
      jnp.logical_and(anderson_freq, is_not_init),
      perform_anderson_step,  # extrapolation
      use_param,  # re-use previous iterate instead
      operand=(params, state)
    )

    params_history = state.params_history
    residuals_history = state.residuals_history
    residual_gram = state.residual_gram
    pos = jnp.mod(state.iter_num, self.history_size)

    next_params, aux = self._value_with_aux(extrapolated, *args, **kwargs)

    residual = tree_sub(next_params, extrapolated)
    ret = update_history(pos, params_history, residuals_history,
                         residual_gram, extrapolated, residual)
    params_history, residuals_history, residual_gram, error = ret

    next_state = AndersonState(iter_num=state.iter_num+1,
                               error=error,
                               params_history=params_history,
                               residuals_history=residuals_history,
                               residual_gram=residual_gram,
                               aux=aux,
                               num_fun_eval=state.num_fun_eval+1)

    if self.verbose:
      self.log_info(next_state, error_name="Residual Norm")
    return base.OptStep(params=next_params, state=next_state)

  def optimality_fun(self, params, *args, **kwargs):
    """Optimality function mapping compatible with ``@custom_root``."""
    next_params, _ = self._value_with_aux(params, *args, **kwargs)
    return tree_sub(next_params, params)

  def __post_init__(self):
    super().__post_init__()

    if self.history_size < 2:
      raise ValueError("history_size should be greater or equal to 2.")

    if self.has_aux:
      fun_ = self.fixed_point_fun
    else:
      fun_ = lambda p, *a, **kw: (self.fixed_point_fun(p, *a, **kw), None)

    self._value_with_aux = fun_

    self.reference_signature = self.fixed_point_fun
