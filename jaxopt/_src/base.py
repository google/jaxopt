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
import dataclasses
import functools
import itertools

from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Optional
from typing import Union
from typing import Tuple

import jax
import jax.numpy as jnp

from jaxopt import implicit_diff as idf

# Do not use jaxopt.linear_solve to avoid circular imports: use
# jaxopt._src.linear_solve instead.
# This allows to define linear solver with base.Solver interface,
# and then exporting them in jaxopt.linear_solve.
from jaxopt._src import linear_solve

from jaxopt import loop
from jaxopt import tree_util


AutoOrBoolean = Union[str, bool]
ArrayPair = Tuple[jnp.ndarray, jnp.ndarray]

NUM_EVAL_DTYPE = 'int32'

class OptStep(NamedTuple):
  params: Any
  state: Any


class KKTSolution(NamedTuple):
  primal: Any
  dual_eq: Optional[Any] = None
  dual_ineq: Optional[Any] = None


# pylint: disable=g-bare-generic
def _add_aux_to_value_and_grad(value_and_grad: Callable) -> Callable:
  def value_and_grad_with_aux(*a, **kw):
    v, g = value_and_grad(*a, **kw)
    return (v, None), g

  return value_and_grad_with_aux


def _add_aux_to_fun(fun: Callable) -> Callable:
  def fun_with_aux(*a, **kw):
    return fun(*a, **kw), None

  return fun_with_aux


def _split_value_and_grad_with_aux(
    value_and_grad_with_aux: Callable,
) -> Callable:
  def fun_with_aux(*a, **kw):
    (v, aux), _ = value_and_grad_with_aux(*a, **kw)
    return (v, aux)

  def grad_with_aux(*a, **kw):
    (_, aux), g = value_and_grad_with_aux(*a, **kw)
    return (g, aux)

  return fun_with_aux, grad_with_aux


def _remove_aux_from_value_and_grad(
    value_and_grad_with_aux: Callable,
) -> Tuple[Callable, Callable, Callable]:
  def grad_without_aux(*a, **kw):
    (_, aux), g = value_and_grad_with_aux(*a, **kw)
    return g

  def fun_without_aux(*a, **kw):
    (v, aux), g = value_and_grad_with_aux(*a, **kw)
    return v

  def value_and_grad_without_aux(*a, **kw):
    (v, aux), g = value_and_grad_with_aux(*a, **kw)
    return (v, g)

  return fun_without_aux, grad_without_aux, value_and_grad_without_aux


def _make_funs_without_aux(
    fun: Callable,
    value_and_grad: Union[bool, Callable],
    has_aux: bool,
) -> Tuple[Callable, Callable, Callable]:
  """Creates fun, grad_fun and value_and_grad_fun functions without aux."""
  if isinstance(value_and_grad, bool) and value_and_grad:
    # Case when `fun` is a user-provided `value_and_grad`.
    if has_aux:
      return _remove_aux_from_value_and_grad(fun)
    else:
      return _remove_aux_from_value_and_grad(_add_aux_to_value_and_grad(fun))
  if isinstance(value_and_grad, bool) and not value_and_grad:
    # Case when `fun` is just a scalar-valued function.
    if has_aux:
      return _remove_aux_from_value_and_grad(
          jax.value_and_grad(fun, has_aux=True)
      )
    else:
      value_and_grad_ = jax.value_and_grad(fun)
      grad = lambda *a, **ka: value_and_grad_(*a, **ka)[1]
      return fun, grad, value_and_grad_
  else:
    # Case when `fun` is the value function, and `value_and_grad` returns
    # both value and grad.
    if has_aux:
      fun_ = lambda *a, **ka: fun(*a, **ka)[0]
      _, grad_, value_and_grad_ = _remove_aux_from_value_and_grad(
          value_and_grad
      )
      return fun_, grad_, value_and_grad_
    else:
      grad_ = lambda *a, **ka: value_and_grad(*a, **ka)[1]
      return fun, grad_, value_and_grad


def _make_funs_with_aux(
    fun: Callable,
    value_and_grad: Union[bool, Callable],
    has_aux: bool,
):
  """Creates fun, grad_fun and value_and_grad_fun functions with aux output.

  Args:
    fun:
    value_and_grad: If `value_and_grad` is True, `fun` should return both value
      and gradient. If `value_and_grad` is a Callable, `fun` should return value
      only, and `value_and_grad` returns value and gradient.
    has_aux: If `has_aux` is True, then any Callable arguments should return
      `(value, aux)` tuple in place of just `value`.

  Returns:
    Three callables, fun, grad, and value_and_grad.
    `fun` returns (value, aux) tuple.
    `grad` returns (grad, aux) tuple.
    `value_and_grad` returns ((value, aux), grad) nested tuple.
    If has_aux is False, then all three returned functions return aux=None.
  """
  if isinstance(value_and_grad, bool) and value_and_grad:
    # Case when `fun` is a user-provided `value_and_grad`.
    if has_aux:
      value_and_grad_ = fun
    else:
      value_and_grad_ = _add_aux_to_value_and_grad(fun)
    fun_, grad_ = _split_value_and_grad_with_aux(value_and_grad_)
    return fun_, grad_, value_and_grad_

  if isinstance(value_and_grad, bool) and not value_and_grad:
    # Case when `fun` is just a scalar-valued function.
    if has_aux:
      fun_ = fun
    else:
      fun_ = _add_aux_to_fun(fun)
    value_and_grad_ = jax.value_and_grad(fun_, has_aux=True)
  else:
    # Case when `fun` is the value function, and `value_and_grad` returns
    # both value and grad.
    if has_aux:
      fun_ = fun
      value_and_grad_ = value_and_grad
    else:
      fun_ = _add_aux_to_fun(fun)
      value_and_grad_ = _add_aux_to_value_and_grad(value_and_grad)

  _, grad_ = _split_value_and_grad_with_aux(value_and_grad_)
  return fun_, grad_, value_and_grad_


class Solver(abc.ABC):
  """Base class for solvers.

  A solver should implement a `run` method:

    `params, state = run(init_params, *args, **kwargs)`

  In addition, we will assume that the solver defines a notion of optimality:
    - `pytree = optimality_fun(params, *args, **kwargs)

  Note: subclasses of `Solver` which are also Python dataclasses must disable
  the generation of an `__eq__` method (`eq=False`) for compatibility with
  Python 3.7. Not doing so results in the generation of a `__hash__` method
  that clashes with JAX transformations when applied directly to methods of
  the class. This restriction will be lifted in the future when support for
  Python 3.7 is dropped and / or another solution is found.
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

  def attribute_names(self):
    return tuple(field.name for field in dataclasses.fields(self))

  def attribute_values(self):
    return tuple(getattr(self, name) for name in self.attribute_names())


class IterativeSolver(Solver):
  """Base class for iterative solvers.

  Any iterative solver should implement `init_state` and `update` methods:
    - `state = init_state(init_params, *args, **kwargs)`
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

  def _get_unroll_option(self):
    """Returns unroll option based on user-provided attributes."""
    if self.unroll == "auto":
      # We unroll when implicit diff is disabled or when jit is disabled.
      return not getattr(self, "implicit_diff", True) or not self.jit
    else:
      return self.unroll

  def _cond_fun(self, inputs):
    _, state = inputs[0]
    return state.error > self.tol
  
  def log_info(self, state, error_name='Error', additional_info={}):
    """Base info at the end of the update."""
    other_info_kw = ' '.join([key + ":{} " for key in additional_info.keys()])
    name = self.__class__.__name__
    jax.debug.print(
      "INFO: jaxopt." + name + ": " + \
      "Iter: {} " + \
      error_name + " (stop. crit.): {} " + \
      other_info_kw,
      state.iter_num,
      state.error,
      *additional_info.values(),
      ordered=True
    )

  def _body_fun(self, inputs):
    (params, state), (args, kwargs) = inputs
    return self.update(params, state, *args, **kwargs), (args, kwargs)

  # TODO(frostig,mblondel): temporary workaround to accommodate line
  # search as an iterative solver, but for this reason and others
  # (automatic implicit diff) we should consider having it not be one.
  def _make_zero_step(self, init_params, state) -> OptStep:
    if isinstance(init_params, OptStep):
      return OptStep(params=init_params.params, state=state)
    else:
      return OptStep(params=init_params, state=state)

  def _run(self,
           init_params: Any,
           *args,
           **kwargs) -> OptStep:
    state = self.init_state(init_params, *args, **kwargs)

    # We unroll the very first iteration. This allows `init_val` and `body_fun`
    # below to have the same output type, which is a requirement of
    # lax.while_loop and lax.scan.
    #
    # TODO(frostig,mblondel): if we could check concreteness of self.maxiter,
    # and we knew that it is concrete here, then we could optimize away the
    # redundant first step, e.g.:
    #
    #   maxiter = get_maybe_concrete(self.maxiter)  # concrete value or None
    #   if maxiter == 0:
    #     return OptStep(params=init_params, state=state)
    #
    # In the general case below, we prefer to use `jnp.where` instead
    # of a `lax.cond` for now in order to avoid staging the initial
    # update and the run loop. They might not be staging compatible.

    zero_step = self._make_zero_step(init_params, state)

    opt_step = self.update(init_params, state, *args, **kwargs)
    init_val = (opt_step, (args, kwargs))

    unroll = self._get_unroll_option()

    many_step = loop.while_loop(
        cond_fun=self._cond_fun, body_fun=self._body_fun,
        init_val=init_val, maxiter=self.maxiter - 1, jit=self.jit,
        unroll=unroll)[0]

    return tree_util.tree_map(
        functools.partial(_where, self.maxiter == 0), zero_step, many_step,
        is_leaf=lambda x: x is None)  # state attributes can sometimes be None

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
      reference_signature = getattr(self, "reference_signature", None)
      decorator = idf.custom_root(
          self.optimality_fun,
          has_aux=True,
          solve=self.implicit_diff_solve,
          reference_signature=reference_signature)
      run = decorator(run)

    return run(init_params, *args, **kwargs)

  def __post_init__(self):
    if self.jit:
      self.update = jax.jit(self.update)


def _where(cond, x, y):
  if x is None: return y
  if y is None: return x
  return jnp.where(cond, x, y)


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
    # Some initializations need the data so we need to draw a batch from the
    # iterator.
    data = next(iterator)
    state = self.init_state(init_params, *args, **kwargs, data=data)
    params = init_params

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


class LineSearchStep(NamedTuple):
  stepsize: float
  state: Any


class IterativeLineSearch(IterativeSolver):

  def _make_zero_step(self, init_stepsize, state) -> LineSearchStep:
    return LineSearchStep(stepsize=init_stepsize, state=state)

  def run(self,
          init_stepsize: float,
          params: Any,
          value: Optional[float] = None,
          grad: Optional[Any] = None,
          descent_direction: Optional[Any] = None,
          fun_args: list = [],
          fun_kwargs: dict = {}) -> LineSearchStep:

    return super()._run(init_stepsize, params, value, grad, descent_direction,
                        fun_args, fun_kwargs)
