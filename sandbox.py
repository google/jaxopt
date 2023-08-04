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
# TODO(pedregosa): add a 'maxiter' and 'callback' keyword option for all wrappers,
#   currently only ScipyMinimize exposes this option.
"""

import abc
import dataclasses
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
from jax.config import config
import jax.numpy as jnp
import jax.tree_util as tree_util
from jax.tree_util import register_pytree_node_class
from jaxopt._src import base
from jaxopt._src import implicit_diff as idf
from jaxopt._src import projection
from jaxopt._src.tree_util import tree_sub
import numpy as onp
import scipy as osp
from scipy.optimize import LbfgsInvHessProduct


@register_pytree_node_class
class LbfgsInvHessProductPyTree(LbfgsInvHessProduct):
  """
  Registers the LbfgsInvHessProduct object as a PyTree.
  This object is typically returned by the L-BFSG-B optimizer to efficiently
  store the inverse of the Hessian matrix evaluated at the best-fit parameters.
  """

  def __init__(self, sk, yk):
    """
    Construct the operator.
    This is the same constructor as the original LbfgsInvHessProduct class,
    except that numpy has been replaced by jax.numpy and no call to the
    numpy.ndarray constuctor is performed.
    """
    if sk.shape != yk.shape or sk.ndim != 2:
      raise ValueError('sk and yk must have matching shape, (n_corrs, n)')
    n_corrs, n = sk.shape
    self.dtype = jnp.float64 if config.jax_enable_x64 is True else jnp.float32
    self.shape = (n, n)
    self.sk = sk
    self.yk = yk
    self.n_corrs = n_corrs
    self.rho = 1 / jnp.einsum('ij,ij->i', sk, yk)


  def __repr__(self):
      return "LbfgsInvHessProduct(sk={}, yk={})".format(self.sk, self.yk)

  def tree_flatten(self):
      children = (self.sk, self.yk)
      aux_data = None
      return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
      return cls(*children)


class ScipyMinimizeInfo(NamedTuple):
  """Named tuple with results for `scipy.optimize.minimize` wrappers."""
  fun_val: jnp.ndarray
  success: bool
  status: int
  iter_num: int
  hess_inv: Optional[Union[jnp.ndarray, LbfgsInvHessProductPyTree]]
  num_fun_eval: int = 0
  num_jac_eval: int = 0
  num_hess_eval: int = 0


class ScipyRootInfo(NamedTuple):
  """Named tuple with results for `scipy.optimize.root` wrappers."""
  fun_val: float
  success: bool
  status: int
  iter_num: int

  num_fun_eval: int = 0


class ScipyLeastSquaresInfo(NamedTuple):
  """Named tuple with results for `scipy.optimize.least_squares` wrappers."""
  cost_val: float
  fun_val: jnp.ndarray
  success: bool
  status: int
  num_fun_eval: int
  num_jac_eval: Optional[int]
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
    maxiter: Maximum number of iterations to perform. Depending on the method,
      each iteration may use several function evaluations.
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
  optimality_fun: Callable
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

    def scipy_fun(x_onp: onp.ndarray, scipy_args: Any) -> Tuple[onp.ndarray, onp.ndarray]:
      # scipy_args is unused but must appear in the signature since
      # the `args` argument passed to osp.optimize.root is not None.
      del scipy_args  # unused
      x_jnp = onp_to_jnp(x_onp)
      value_jnp = self.optimality_fun(x_jnp, *args, **kwargs)
      jacs_jnp = self._jac_fun(x_jnp, *args, **kwargs)
      return jnp_to_onp(value_jnp, self.dtype), jac_jnp_to_onp(jacs_jnp)

    params_dtype = jnp.asarray(init_params).dtype
    
    def scipy_call(init_x):
      # Argument `args` is unused but must be not None to ensure that some sanity checks are performed
      # correctly in Scipy for optimizers that don't use the Jacobian (such as Broyden).
      # See the related issue: https://github.com/google/jaxopt/issues/290
      res = osp.optimize.root(scipy_fun,
                              init_x,
                              args=(None,),
                              jac=True,
                              tol=self.tol,
                              method=self.method,
                              options=self.options)
      params = jnp.asarray(res.x, dtype=params_dtype)
      fun, success, status = jnp.array(res.fun, dtype=params_dtype), jnp.array(res.success, onp.bool_), jnp.array(res.status, onp.int32)
      return params, fun, success, status

    params_shape_dtype = jax.ShapeDtypeStruct(shape=init_params.shape , dtype=params_dtype)
    # NOTE(vroulet): may record the function out dtype for better compatibility
    fun_shape_dtype = jax.ShapeDtypeStruct(shape=init_params.shape, dtype=params_dtype)
    succes_shape_dtype = jax.ShapeDtypeStruct(shape=[], dtype=jnp.bool_)
    status_shape_dtype = jax.ShapeDtypeStruct(shape=[], dtype=jnp.int32)

    shapes_dtypes = (params_shape_dtype, fun_shape_dtype, succes_shape_dtype, status_shape_dtype)
    params, fun, success, status = jax.pure_callback(scipy_call, shapes_dtypes, init_params)
    
    info = ScipyRootInfo(fun_val=jnp.asarray(fun),
                         success=jnp.asarray(success),
                         status=jnp.asarray(status),
                         iter_num=None, # NOTE(vroulet) to fix later
                         num_fun_eval=None) # NOTE(vroulet) to fix later
    
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

import jax
import jax.numpy as jnp
import numpy as np

def opt_fun(x, b):
    return x**2-b

x = np.random.randn(3)
b = np.random.randn(3)**2

root_finder  = ScipyRootFinding('hybr', optimality_fun=opt_fun)
y = root_finder.run(x,b).params
print(y)
print(jnp.sqrt(b))

# x = np.random.randn(2, 3)
# b = np.random.randn(2, 3)**2

# root_finder  = ScipyRootFinding('hybr', optimality_fun=opt_fun)
# jax.vmap(root_finder.run, in_axes=(0,0))(x, b)
