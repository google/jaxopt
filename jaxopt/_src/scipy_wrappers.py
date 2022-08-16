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

"""Wraps SciPy's optimization routines with PyTree and implicit diff support.

# TODO(fllinares): add support for `LinearConstraint`s.
# TODO(fllinares): add support for methods requiring Hessian / Hessian prods.
# TODO(fllinares): possibly hardcode `dtype` attribute, as likely useless.
"""

import abc
from dataclasses import dataclass

from typing import Any
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import projection
from jaxopt._src.tree_util import tree_sub

import numpy as onp
import scipy as osp


class ScipyMinimizeInfo(NamedTuple):
  """Named tuple with results for `scipy.optimize.minimize` wrappers."""
  fun_val: jnp.ndarray
  success: bool
  status: int
  iter_num: int


class ScipyRootInfo(NamedTuple):
  """Named tuple with results for `scipy.optimize.root` wrappers."""
  fun_val: float
  success: bool
  status: int


class ScipyLeastSquaresInfo(NamedTuple):
  """Named tuple with results for `scipy.optimize.least_squares` wrappers."""
  cost_val: float
  fun_val: jnp.ndarray
  success: bool
  status: int
  nfev: int
  njev: Optional[int]
  error: float


class PyTreeTopology(NamedTuple):
  """Stores info to reconstruct PyTree from flattened PyTree leaves.

  # TODO(fllinares): more specific type annotations for attributes?

  Attributes:
    treedef: the PyTreeDef object encoding the structure of the target PyTree.
    shapes: an iterable with the shapes of each leaf in the target PyTree.
    dtypes: an iterable with the dtypes of each leaf in the target PyTree.
    sizes: an iterable with the sizes of each leaf in the target PyTree.
    n_leaves: the number of leaves in the target PyTree.
  """
  treedef: Any
  shapes: Sequence[Any]
  dtypes: Sequence[Any]

  @property
  def sizes(self):
    return [int(onp.prod(shape)) for shape in self.shapes]

  @property
  def n_leaves(self):
    return len(self.shapes)


def jnp_to_onp(x_jnp: Any,
               dtype: Optional[Any] = onp.float64) -> onp.ndarray:
  """Converts JAX PyTree into repr suitable for scipy.optimize.minimize.

  Several of SciPy's optimization routines require inputs and/or outputs to be
  onp.ndarray<float>[n]. Given an input PyTree `x_jnp`, this function will
  flatten all its leaves and, if there is more than one leaf, the corresponding
  flattened arrays will be concatenated and, optionally, casted to `dtype`.

  Args:
    x_jnp: a PyTree of jnp.ndarray with structure identical to init.
    dtype: if not None, ensure output is a NumPy array of this dtype.
  Returns:
    A single onp.ndarray<dtype>[n] array, consisting of all leaves of x_jnp
    flattened and concatenated. If dtype is None, the output dtype will be
    determined by NumPy's casting rules for the concatenate method.
  """
  x_onp = [onp.asarray(leaf, dtype).reshape(-1)
           for leaf in tree_util.tree_leaves(x_jnp)]
  # NOTE(fllinares): return value must *not* be read-only, I believe.
  return onp.concatenate(x_onp)


def make_jac_jnp_to_onp(input_pytree_topology: PyTreeTopology,
                        output_pytree_topology: PyTreeTopology,
                        dtype: Optional[Any] = onp.float64) -> Callable:
  """Returns function "flattening" Jacobian for given in/out PyTree topologies.

  For a smooth function `fun(x_jnp, *args, **kwargs)` taking an arbitrary
  PyTree `x_jnp` as input and returning another arbitrary PyTree `y_jnp` as
  output, JAX's transforms such as `jax.jacrev` or `jax.jacfwd` will return a
  Jacobian with a PyTree structure reflecting the input and output PyTrees.
  However, several of SciPy's optimization routines expect inputs and outputs to
  be 1D NumPy arrays and, thus, Jacobians to be 2D NumPy arrays.

  Given the Jacobian of `fun(x_jnp, *args, **kwargs)` as provided by JAX,
  `jac_jnp_to_onp` will format it to match the Jacobian of
  `jnp_to_onp(fun(x_jnp, *args, **kwargs))` w.r.t. `jnp_to_onp(x_jnp)`,
  where `jnp_to_onp` is a vectorization operator for arbitrary PyTrees.

  Args:
    input_pytree_topology: a PyTreeTopology encoding the topology of the input
      PyTree.
    output_pytree_topology: a PyTreeTopology encoding the topology of the output
      PyTree.
    dtype: if not None, ensure output is a NumPy array of this dtype.
  Returns:
    A function "flattening" Jacobian for given input and output PyTree
    topologies.
  """
  ravel_index = lambda i, j: j + i * input_pytree_topology.n_leaves

  def jac_jnp_to_onp(jac_pytree: Any):
    # Builds flattened Jacobian blocks such that `jacs_onp[i][j]` equals the
    # Jacobian of vec(i-th leaf of output_pytree) w.r.t.
    # vec(j-th leaf of input_pytree), where vec() is the vectorization op.,
    # i.e. reshape(input, [-1]).
    jacs_leaves = tree_util.tree_leaves(jac_pytree)
    jacs_onp = []
    for i, output_size in enumerate(output_pytree_topology.sizes):
      jacs_onp_i = []
      for j, input_size in enumerate(input_pytree_topology.sizes):
        jac_leaf = onp.asarray(jacs_leaves[ravel_index(i, j)], dtype)
        jac_leaf = jac_leaf.reshape([output_size, input_size])
        jacs_onp_i.append(jac_leaf)
      jacs_onp.append(jacs_onp_i)
    return onp.block(jacs_onp)

  return jac_jnp_to_onp


def make_onp_to_jnp(pytree_topology: PyTreeTopology) -> Callable:
  """Returns inverse of `jnp_to_onp` for a specific PyTree topology.

  Args:
    pytree_topology: a PyTreeTopology encoding the topology of the original
      PyTree to be reconstructed.
  Returns:
    The inverse of `jnp_to_onp` for a specific PyTree topology.
  """
  treedef, shapes, dtypes = pytree_topology
  split_indices = onp.cumsum(list(pytree_topology.sizes[:-1]))
  def onp_to_jnp(x_onp: onp.ndarray) -> Any:
    """Inverts `jnp_to_onp` for a specific PyTree topology."""
    flattened_leaves = onp.split(x_onp, split_indices)
    x_jnp = [jnp.asarray(leaf.reshape(shape), dtype)
             for leaf, shape, dtype in zip(flattened_leaves, shapes, dtypes)]
    return tree_util.tree_unflatten(treedef, x_jnp)
  return onp_to_jnp


def pytree_topology_from_example(x_jnp: Any) -> PyTreeTopology:
  """Returns a PyTreeTopology encoding the PyTree structure of `x_jnp`."""
  leaves, treedef = tree_util.tree_flatten(x_jnp)
  shapes = [jnp.asarray(leaf).shape for leaf in leaves]
  dtypes = [jnp.asarray(leaf).dtype for leaf in leaves]
  return PyTreeTopology(treedef=treedef, shapes=shapes, dtypes=dtypes)


@dataclass(eq=False)
class ScipyWrapper(base.Solver):
  """Wraps over `scipy.optimize` methods with PyTree and implicit diff support.

  Attributes:
    method: the `method` argument for `scipy.optimize`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `fun(...)[0]` is the
      objective.
  """
  method: Optional[str] = None
  dtype: Optional[Any] = onp.float64
  jit: bool = True
  implicit_diff_solve: Optional[Callable] = None
  has_aux: bool = False

  def optimality_fun(self, sol, *args, **kwargs):
    raise NotImplementedError(
        'ScipyWrapper subclasses must implement `optimality_fun` as needed.')

  def __post_init__(self):
    # Set up implicit diff.
    decorator = idf.custom_root(self.optimality_fun,
                                has_aux=True,
                                solve=self.implicit_diff_solve)
    # pylint: disable=g-missing-from-attributes
    self.run = decorator(self.run)


@dataclass(eq=False)
class ScipyMinimize(ScipyWrapper):
  """`scipy.optimize.minimize` wrapper

  This wrapper is for unconstrained minimization only.
  It supports pytrees and implicit diff.

  Attributes:
    fun: a smooth function of the form `fun(x, *args, **kwargs)`.
    method: the `method` argument for `scipy.optimize.minimize`.
      Should be one of
        * 'Nelder-Mead'
        * 'Powell'
        * 'CG'
        * 'BFGS'
        * 'Newton-CG'
        * 'L-BFGS-B'
        * 'TNC'
        * 'COBYLA'
        * 'SLSQP'
        * 'trust-constr'
        * 'dogleg'
        * 'trust-ncg'
        * 'trust-exact'
        * 'trust-krylov'
    tol: the `tol` argument for `scipy.optimize.minimize`.
    options: the `options` argument for `scipy.optimize.minimize`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `fun(...)[0]` is the
      objective.
  """
  fun: Callable = None
  tol: Optional[float] = None
  options: Optional[Dict[str, Any]] = None

  def optimality_fun(self, sol, *args, **kwargs):
    """Optimality function mapping compatible with `@custom_root`."""
    return self._grad_fun(sol, *args, **kwargs)

  def _run(self, init_params, bounds, *args, **kwargs):
    """Wraps `scipy.optimize.minimize`."""
    # Sets up the "JAX-SciPy" bridge.
    pytree_topology = pytree_topology_from_example(init_params)
    onp_to_jnp = make_onp_to_jnp(pytree_topology)

    def scipy_fun(x_onp: onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray]:
      x_jnp = onp_to_jnp(x_onp)
      value, grads = self._value_and_grad_fun(x_jnp, *args, **kwargs)
      return onp.asarray(value, self.dtype), jnp_to_onp(grads, self.dtype)

    if bounds is not None:
      bounds = osp.optimize.Bounds(lb=jnp_to_onp(bounds[0], self.dtype),
                                   ub=jnp_to_onp(bounds[1], self.dtype))

    res = osp.optimize.minimize(scipy_fun, jnp_to_onp(init_params, self.dtype),
                                jac=True,
                                tol=self.tol,
                                bounds=bounds,
                                method=self.method,
                                options=self.options)

    params = tree_util.tree_map(jnp.asarray, onp_to_jnp(res.x))
    info = ScipyMinimizeInfo(fun_val=jnp.asarray(res.fun),
                             success=res.success,
                             status=res.status,
                             iter_num=res.nit)
    return base.OptStep(params, info)

  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> base.OptStep:
    """Runs the solver.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to `fun`.
      **kwargs: additional keyword arguments to be passed to `fun`.
    Returns:
      (params, info).
    """
    return self._run(init_params, None, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()

    if self.has_aux:
      self.fun = lambda x, *args, **kwargs: self.fun(x, *args, **kwargs)[0]

    # Pre-compile useful functions.
    self._grad_fun = jax.grad(self.fun)
    self._value_and_grad_fun = jax.value_and_grad(self.fun)
    if self.jit:
      self._grad_fun = jax.jit(self._grad_fun)
      self._value_and_grad_fun = jax.jit(self._value_and_grad_fun)


@dataclass(eq=False)
class ScipyBoundedMinimize(ScipyMinimize):
  """`scipy.optimize.minimize` wrapper.

  This wrapper is for minimization subject to box constraints only.

  Attributes:
    fun: a smooth function of the form `fun(x, *args, **kwargs)`.
    method: the `method` argument for `scipy.optimize.minimize`.
    tol: the `tol` argument for `scipy.optimize.minimize`.
    options: the `options` argument for `scipy.optimize.minimize`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `fun(...)[0]` is the
      objective.
  """

  def _fixed_point_fun(self, sol, bounds, args, kwargs):
    step = tree_sub(sol, self._grad_fun(sol, *args, **kwargs))
    return projection.projection_box(step, bounds)

  def optimality_fun(self, sol, bounds, *args, **kwargs):
    """Optimality function mapping compatible with `@custom_root`."""
    fp = self._fixed_point_fun(sol, bounds, args, kwargs)
    return tree_sub(fp, sol)

  def run(self,
          init_params: Any,
          bounds: Optional[Any],
          *args,
          **kwargs) -> base.OptStep:
    """Runs the solver.

    Args:
      init_params: pytree containing the initial parameters.
      bounds: an optional tuple `(lb, ub)` of pytrees with structure identical
        to `init_params`, representing box constraints.
      *args: additional positional arguments to be passed to `fun`.
      **kwargs: additional keyword arguments to be passed to `fun`.
    Returns:
      (params, info).
    """
    return self._run(init_params, bounds, *args, **kwargs)


@dataclass(eq=False)
class ScipyRootFinding(ScipyWrapper):
  """`scipy.optimize.root` wrapper.

  It supports pytrees and implicit diff.

  Attributes:
    optimality_fun: a smooth vector function of the form
      `optimality_fun(x, *args, **kwargs)` whose root is to be found. It must
      return as output a PyTree with structure identical to x.
    method: the `method` argument for `scipy.optimize.root`.
      Should be one of
        * 'hybr'
        * 'lm'
        * 'broyden1'
        * 'broyden2'
        * 'anderson'
        * 'linearmixing'
        * 'diagbroyden'
        * 'excitingmixing'
        * 'krylov'
        * 'df-sane'
    tol: the `tol` argument for `scipy.optimize.root`.
    options: the `options` argument for `scipy.optimize.root`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `optimality_fun(...)[0]` is
      the optimality function.
    use_jacrev: whether to compute the Jacobian of `optimality_fun` using
      `jax.jacrev` (True) or `jax.jacfwd` (False).
  """
  optimality_fun: Callable = None
  tol: Optional[float] = None
  options: Optional[Dict[str, Any]] = None
  use_jacrev: bool = True

  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> base.OptStep:
    """Runs the solver.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to `fun`.
      **kwargs: additional keyword arguments to be passed to `fun`.
    Returns:
      (params, info).
    """
    # Sets up the "JAX-SciPy" bridge.
    pytree_topology = pytree_topology_from_example(init_params)
    onp_to_jnp = make_onp_to_jnp(pytree_topology)
    jac_jnp_to_onp = make_jac_jnp_to_onp(pytree_topology,
                                         pytree_topology,
                                         self.dtype)

    def scipy_fun(x_onp: onp.ndarray) -> Tuple[onp.ndarray, onp.ndarray]:
      x_jnp = onp_to_jnp(x_onp)
      value_jnp = self.optimality_fun(x_jnp, *args, **kwargs)
      jacs_jnp = self._jac_fun(x_jnp, *args, **kwargs)
      return jnp_to_onp(value_jnp, self.dtype), jac_jnp_to_onp(jacs_jnp)

    res = osp.optimize.root(scipy_fun, jnp_to_onp(init_params, self.dtype),
                            jac=True,
                            tol=self.tol,
                            method=self.method,
                            options=self.options)

    params = tree_util.tree_map(jnp.asarray, onp_to_jnp(res.x))
    info = ScipyRootInfo(fun_val=jnp.asarray(res.fun),
                         success=res.success,
                         status=res.status)
    return base.OptStep(params, info)

  def __post_init__(self):
    super().__post_init__()

    if self.has_aux:
      def optimality_fun(x, *args, **kwargs):
        return self.optimality_fun(x, *args, **kwargs)[0]
      self.optimality_fun = optimality_fun

    # Pre-compile useful functions.
    self._jac_fun = (jax.jacrev(self.optimality_fun) if self.use_jacrev
                     else jax.jacfwd(self.optimality_fun))
    if self.jit:
      self.optimality_fun = jax.jit(self.optimality_fun)
      self._jac_fun = jax.jit(self._jac_fun)


# NOTE: relative to `scipy.optimize.least_squares`, the functions below absorb
# the squaring of residuals to avoid numerical issues for the gradient of the
# Huber loss at 0.
LS_RHO_FUNS = {
    'linear': lambda z: z ** 2,
    'soft_l1': lambda z: 2.0 * ((1 + z ** 2) ** 0.5 - 1),
    'huber': lambda z: jnp.where(z <= 1, z ** 2, 2.0 * z - 1),
    'cauchy': lambda z: jnp.log1p(z ** 2),
    'arctan': lambda z: jnp.arctan(z ** 2),
}
LS_DEFAULT_OPTIONS = {
    'ftol': 1e-8,  # float
    'xtol': 1e-8,  # float
    'gtol': 1e-8,  # float
    'x_scale': 1.0,  # Any
    'f_scale': 1.0,  # float
    'tr_solver': None,  # Optional[str]
    'tr_options': {},  # Optional[Dict[str, Any]]
    'max_nfev': None,  # Optional[int],
}


@dataclass(eq=False)
class ScipyLeastSquares(ScipyWrapper):
  """Wraps over `scipy.optimize.least_squares` with PyTree & imp. diff support.

  This solver minimizes::
    0.5 * sum(loss(fun(x, *args, **kwargs) ** 2)).
  This wrapper is for unconstrained minimization only.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)`` computing the
      residuals from the model parameters `x`.
    loss: the `loss` argument for `scipy.optimize.least_squares`. However,
      arbitrary losses specified with a Callable are not yet supported.
    options: additional kwargs for `scipy.optimize.least_squares`.
    method: the `method` argument for `scipy.optimize.least_squares`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `fun(...)[0]` are the
      residuals.
    use_jacrev: whether to compute the Jacobian of `fun` using `jax.jacrev`
      (True) or `jax.jacfwd` (False).
  """
  fun: Callable = None
  loss: str = 'linear'
  options: Optional[Dict[str, Any]] = None
  use_jacrev: bool = True

  def _cost_fun(self, params, *args, **kwargs):
    residuals = self.fun(params, *args, **kwargs)
    # NOTE: `self._rho` includes the squaring of residuals in its definition.
    losses = self._rho(residuals / self.options['f_scale'])
    return 0.5 * jnp.square(self.options['f_scale']) * jnp.mean(losses)

  def optimality_fun(self, sol, *args, **kwargs):
    """Optimality function mapping compatible with `@custom_root`."""
    return self._grad_cost_fun(sol, *args, **kwargs)

  def _run(self, init_params, bounds, *args, **kwargs):
    """Wraps `scipy.optimize.least_squares`."""
    # Sets up the "JAX-SciPy" bridge.
    init_output = self.fun(init_params, *args, **kwargs)
    input_pytree_topology = pytree_topology_from_example(init_params)
    output_pytree_topology = pytree_topology_from_example(init_output)
    onp_to_jnp = make_onp_to_jnp(input_pytree_topology)
    jac_jnp_to_onp = make_jac_jnp_to_onp(input_pytree_topology,
                                         output_pytree_topology,
                                         self.dtype)

    def scipy_fun(x_onp: onp.ndarray) -> onp.ndarray:
      x_jnp = onp_to_jnp(x_onp)
      value_jnp = self.fun(x_jnp, *args, **kwargs)
      return jnp_to_onp(value_jnp, self.dtype)

    def scipy_jac(x_onp: onp.ndarray) -> onp.ndarray:
      x_jnp = onp_to_jnp(x_onp)
      jacs_jnp = self._jac_fun(x_jnp, *args, **kwargs)
      return jac_jnp_to_onp(jacs_jnp)

    if bounds is not None:
      bounds = (jnp_to_onp(bounds[0], self.dtype),
                jnp_to_onp(bounds[1], self.dtype))
    else:
      bounds = (-onp.inf, onp.inf)

    res = osp.optimize.least_squares(scipy_fun,
                                     jnp_to_onp(init_params, self.dtype),
                                     jac=scipy_jac,
                                     bounds=bounds,
                                     method=self.method,
                                     loss=self.loss,
                                     **self.options)

    params = tree_util.tree_map(jnp.asarray, onp_to_jnp(res.x))
    info = ScipyLeastSquaresInfo(cost_val=jnp.asarray(res.cost),
                                 fun_val=jnp.asarray(res.fun),
                                 success=res.success,
                                 status=res.status,
                                 nfev=res.nfev,
                                 njev=res.njev,
                                 error=jnp.asarray(res.optimality))
    return base.OptStep(params, info)

  def run(self,
          init_params: Any,
          *args,
          **kwargs) -> base.OptStep:
    """Runs the solver.

    Args:
      init_params: pytree containing the initial parameters.
      *args: additional positional arguments to be passed to `fun`.
      **kwargs: additional keyword arguments to be passed to `fun`.
    Returns:
      (params, info).
    """
    return self._run(init_params, None, *args, **kwargs)

  def __post_init__(self):
    super().__post_init__()

    if self.options is None:
      self.options = LS_DEFAULT_OPTIONS
    else:
      for k, v in LS_DEFAULT_OPTIONS.items():
        if k not in self.options:
          self.options[k] = v

    if self.has_aux:
      self.fun = lambda x, *args, **kwargs: self.fun(x, *args, **kwargs)[0]

    # Handles PyTree inputs for `x_scale` arg.
    if self.options['x_scale'] != 'jac' and not isinstance(
        self.options['x_scale'], float):
      self.options['x_scale'] = jnp_to_onp(self.options['x_scale'], self.dtype)

    # Pre-compile useful functions.
    if self.loss not in LS_RHO_FUNS:
      raise ValueError(f'`loss` must be one of {LS_RHO_FUNS.keys()}.')
    self._rho = LS_RHO_FUNS[self.loss]
    self._jac_fun = (jax.jacrev(self.fun) if self.use_jacrev
                     else jax.jacfwd(self.fun))
    self._grad_cost_fun = jax.grad(self._cost_fun)
    if self.jit:
      self.fun = jax.jit(self.fun)
      self._rho = jax.jit(self._rho)
      self._jac_fun = jax.jit(self._jac_fun)
      self._grad_cost_fun = jax.jit(self._grad_cost_fun)


@dataclass(eq=False)
class ScipyBoundedLeastSquares(ScipyLeastSquares):
  """Wraps over `scipy.optimize.least_squares` with PyTree & imp. diff support.

  This solver minimizes::
    0.5 * sum(loss(fun(x, *args, **kwargs) ** 2))
      subject to bounds[0] <= x <= bounds[1].
  This wrapper is for minimization subject to box constraints only.

  Attributes:
    fun: a smooth function of the form ``fun(x, *args, **kwargs)`` computing the
      residuals from the model parameters `x`.
    loss: the `loss` argument for `scipy.optimize.least_squares`. However,
      arbitrary losses specified with a Callable are not yet supported.
    options: additional kwargs for `scipy.optimize.least_squares`.
    method: the `method` argument for `scipy.optimize.least_squares`.
    dtype: if not None, cast all NumPy arrays to this dtype. Note that some
      methods relying on FORTRAN code, such as the `L-BFGS-B` solver for
      `scipy.optimize.minimize`, require casting to float64.
    jit: whether to JIT-compile JAX-based values and grad evals.
    implicit_diff_solve: the linear system solver to use.
    has_aux: whether function `fun` outputs one (False) or more values (True).
      When True it will be assumed by default that `fun(...)[0]` are the
      residuals.
    use_jacrev: whether to compute the Jacobian of `fun` using `jax.jacrev`
      (True) or `jax.jacfwd` (False).
  """

  def _fixed_point_fun(self, sol, bounds, args, kwargs):
    step = tree_sub(sol, self._grad_cost_fun(sol, *args, **kwargs))
    return projection.projection_box(step, bounds)

  def optimality_fun(self, sol, bounds, *args, **kwargs):
    """Optimality function mapping compatible with `@custom_root`."""
    fp = self._fixed_point_fun(sol, bounds, args, kwargs)
    return tree_sub(fp, sol)

  def run(self,
          init_params: Any,
          bounds: Optional[Any],
          *args,
          **kwargs) -> base.OptStep:
    """Runs the solver.

    Args:
      init_params: pytree containing the initial parameters.
      bounds: an optional tuple `(lb, ub)` of pytrees with structure identical
        to `init_params`, representing box constraints.
      *args: additional positional arguments to be passed to `fun`.
      **kwargs: additional keyword arguments to be passed to `fun`.
    Returns:
      (params, info).
    """
    return self._run(init_params, bounds, *args, **kwargs)
