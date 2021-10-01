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

import abc
import itertools

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff as idf
from jaxopt import linear_solve
from jaxopt import loop
from jaxopt import tree_util


AutoOrBoolean = Union[str, bool]


class OptStep(NamedTuple):
  params: Any
  state: Any


class KKTSolution(NamedTuple):
  primal: Any
  dual_eq: Optional[Any] = None
  dual_ineq: Optional[Any] = None


class Solver(abc.ABC):
  """Base class for solvers.

  A solver should implement a `run` method:

    `params, state = run(init_params, *args, **kwargs)`

  In addition, we will assume that the solver defines a notion of optimality:
    - `pytree = optimality_fun(params, *args, **kwargs)
  """

  @abc.abstractmethod
  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> OptStep:
    pass

  def l2_optimality_error(self, params, *args, **kwargs):
    """Computes the L2 optimality error."""
    optimality = self.optimality_fun(params, *args, **kwargs)
    return tree_util.tree_l2_norm(optimality)


class IterativeSolver(Solver):
  """Base class for iterative solvers.

  Any iterative solver should implement `init` and `update` methods:
    - `params, state = init(init_params, *args, **kwargs)`
    - `next_params, next_state = update(params, state, *args, **kwargs)`

  This class implements a `run` method:

    `params, state = run(init_params, *args, **kwargs)`

  The following attributes are needed by the `run` method:
    - `verbose`
    - `maxiter`
    - `tol`
    - `implicit_diff`
    - `implicit_diff_solve`

  If `implicit_diff` is not present, it is assumed to be True.

  The following attribute is needed in the state:
    - `error`
  """

  def _run(self,
           init_params: Any,
           *args,
           **kwargs) -> OptStep:

    def cond_fun(pair):
      _, state = pair
      if self.verbose:
        print(state.error)
      return state.error > self.tol

    def body_fun(pair):
      params, state = pair
      return self.update(params, state, *args, **kwargs)

    if self.jit == "auto":
      # We always jit unless verbose mode is enabled.
      jit = not self.verbose
    else:
      jit = self.jit

    if self.unroll == "auto":
      # We unroll when implicit diff is disabled or when jit is disabled.
      unroll = not getattr(self, "implicit_diff", True) or not jit
    else:
      unroll = self.unroll

    return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun,
                           init_val=self.init(init_params, *args, **kwargs),
                           maxiter=self.maxiter, jit=jit, unroll=unroll)

  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> OptStep:
    """Runs the optimization loop.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to the update method.
      **kwargs: additional keyword arguments to be passed to the update method.
    Returns:
      (params, state)
    """
    run = self._run

    if getattr(self, "implicit_diff", True):
      decorator = idf.custom_root(self.optimality_fun,
                                  has_aux=True,
                                  solve=self.implicit_diff_solve)
      run = decorator(run)

    return run(init_params, *args, **kwargs)


class StochasticSolver(IterativeSolver):
  """Stochastic solver.

  This class implements a method:

    `params, state = run_iterator(init_params, iterator, *args, **kwargs)`

  The following attribute is needed in the solver:
    - an `maxiter`

  The `update` method must accept a `data` argument for receiving mini-batches
  produced by `iterator`.
  """

  def run_iterator(self,
                   init_params: Any,
                   iterator,
                   *args,
                   **kwargs) -> OptStep:
    """Runs the optimization loop over an iterator.

    Args:
      init_params: pytree containing the initial parameters.
      iterator: iterator generating data batches.
      *args: additional positional arguments to be passed to ``fun``.
      **kwargs: additional keyword arguments to be passed to ``fun``.
    Returns:
      (params, state)
    """
    # TODO(mblondel): data-dependent initialization schemes need a batch.
    params, state = self.init(init_params, *args, **kwargs)

    # TODO(mblondel): try and benchmark lax.fori_loop with host_call for `next`.
    for data in itertools.islice(iterator, 0, self.maxiter):

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
