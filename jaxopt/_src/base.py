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

"""Base definitions useful across the project."""

import itertools

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt import tree_util


class OptStep(NamedTuple):
  params: Any
  state: Any


class KKTSolution(NamedTuple):
  primal: Any
  dual_eq: Optional[Any] = None
  dual_ineq: Optional[Any] = None


class IterativeSolverMixin:
  """Base mixin for iterative batch solvers.

  A batch solver is expected to see the whole dataset on each iteration.
  It stops when convergence is detected using an optimality criterion,
  defined in `optimality_fun`.

  This mixin implements:
    - a method `params, state = run(init_params, *args, **kwargs)`
    - a method `l2_optimality_error`
    - a private method `_set_implicit_diff_run(self)`

  The following is needed from the child solver:
    - a method `params, state = init(init_params, *args, **kwargs)`
    - a method `params, state = update(params, state, *args, **kwargs)
    - a method `optimality_fun(params, *args, **kwargs)
    - an attribute `verbose`
    - an attribute `maxiter`
    - an attribute `tol`
    - an attribute `implicit_diff`

  The following is needed from the state:
    - an attribute `error`
  """

  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> OptStep:
    """Runs the solver until convergence or `maxiter` is reached.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to the update method.
      **kwargs: additional keyword arguments to be passed to the update method.
    Return type:
      OptStep
    Returns:
      (params, state)
    """
    def cond_fun(pair):
      _, state = pair
      if self.verbose:
        print(state.error)
      return state.error > self.tol

    def body_fun(pair):
      params, state = pair
      return self.update(params, state, *args, **kwargs)

    # We always jit unless verbose mode is enabled.
    jit = not self.verbose
    # We unroll when implicit diff is disabled or when jit is disabled.
    unroll = not self.implicit_diff or not jit

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params, *args, **kwargs),
                           maxiter=self.maxiter, jit=jit, unroll=unroll)

  def l2_optimality_error(self, params, *args, **kwargs):
    """Computes the L2 optimality error."""
    optimality = self.optimality_fun(params, *args, **kwargs)
    return tree_util.tree_l2_norm(optimality)

  def _set_implicit_diff_run(self):
    # Set up implicit diff.
    if self.implicit_diff:
      if isinstance(self.implicit_diff, Callable):
        solve = self.implicit_diff
      else:
        solve = linear_solve.solve_normal_cg

      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=solve)
      # pylint: disable=g-missing-from-attributes
      self.run = decorator(self.run)


class StochasticSolverMixin:
  """Implements a run_iterator method for a stochastic solver.

  A stochastic solver is expected to only see a batch on each iteration.
  It does not monitor convergence, apart from checking if `maxiter` is reached.

  The following is needed from the child solver:
    - a method `params, state = init(init_params, *args, **kwargs)`
    - a method `params, state = update(params, state, *args, **kwargs)
    - an optional attribute `pre_update`
    - an attribute `maxiter`

  """

  def run_iterator(self,
                   init_params: Any,
                   iterator,
                   *args,
                   **kwargs) -> OptStep:
    """Runs the solver on a dataset iterator.

    Args:
      init_params: pytree containing the initial parameters.
      iterator: iterator generating data batches.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Return type:
      OptStep
    Returns:
      (params, state)
    """
    # TODO(mblondel): data-dependent initialization schemes need a batch.
    params, state = self.init(init_params, *args, **kwargs)

    # TODO(mblondel): try and benchmark lax.fori_loop with host_call for `next`.
    for data in itertools.islice(iterator, 0, self.maxiter):

      if self.pre_update:
        params, state = self.pre_update(params, state, *args, **kwargs)

      params, state = self.update(params, state, *args, **kwargs, data=data)

    return OptStep(params=params, state=state)


@jax.tree_util.register_pytree_node_class
class LinearOperator(object):

  def __init__(self, A):
    self.A = jnp.array(A)

  def shape(self):
    return self.A.shape

  def matvec(self, x):
    """Computes dot(A, x)."""
    return jnp.dot(self.A, x)

  def matvec_element(self, x, idx):
    """Computes dot(A, x)[idx]."""
    return jnp.dot(self.A[idx], x)

  def rmatvec(self, x):
    """Computes dot(A.T, x)."""
    return jnp.dot(self.A.T, x)

  def rmatvec_element(self, x, idx):
    """Computes dot(A.T, x)[idx]."""
    return jnp.dot(self.A[:, idx], x)

  def update_matvec(self, Ax, delta, idx):
    """Updates dot(A, x) when x[idx] += delta."""
    if len(Ax.shape) == 1:
      return Ax + delta * self.A[:, idx]
    elif len(Ax.shape) == 2:
      return Ax + jnp.outer(self.A[:, idx], delta)
    else:
      raise ValueError("Ax should be a vector or a matrix.")

  def update_rmatvec(self, ATx, delta, idx):
    """Updates dot(A.T, x) when x[idx] += delta."""
    if len(ATx.shape) == 1:
      return ATx + delta * self.A[idx]
    elif len(ATx.shape) == 2:
      raise NotImplementedError
    else:
      raise ValueError("Ax should be a vector or a matrix.")

  def column_l2_norms(self, squared=False):
    ret = jnp.sum(self.A ** 2, axis=0)
    if not squared:
      ret = jnp.sqrt(ret)
    return ret

  def tree_flatten(self):
    return (self.A,), None

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)
