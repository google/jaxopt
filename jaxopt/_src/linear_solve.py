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

"""Linear system solvers."""

from typing import Any
from typing import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from jaxopt._src.tree_util import tree_add_scalar_mul


def _materialize_array(matvec, shape, dtype=None):
  """Materializes the matrix A used in matvec(x) = Ax."""
  x = jnp.zeros(shape, dtype)
  return jax.jacfwd(matvec)(x)


def _make_ridge_matvec(matvec: Callable, ridge: float = 0.0):
  def ridge_matvec(v: Any) -> Any:
    return tree_add_scalar_mul(matvec(v), ridge, v)
  return ridge_matvec


def solve_lu(matvec: Callable, b: jnp.ndarray) -> jnp.ndarray:
  """Solves ``A x = b`` using ``jax.lax.solve``.

  This solver is based on an LU decomposition.
  It will materialize the matrix ``A`` in memory.

  Args:
    matvec: product between ``A`` and a vector.
    b: array.

  Returns:
    array with same structure as ``b``.
  """
  if len(b.shape) == 0:
    return b / _materialize_array(matvec, b.shape)
  elif len(b.shape) == 1:
    A = _materialize_array(matvec, b.shape, b.dtype)
    return jax.numpy.linalg.solve(A, b)
  elif len(b.shape) == 2:
    A = _materialize_array(matvec, b.shape, b.dtype)  # 4d array (tensor)
    A = A.reshape(-1, b.shape[0] * b.shape[1])  # 2d array (matrix)
    return jax.numpy.linalg.solve(A, b.ravel()).reshape(*b.shape)
  else:
    raise NotImplementedError


def solve_cholesky(matvec: Callable, b: jnp.ndarray) -> jnp.ndarray:
  """Solves ``A x = b``, using Cholesky decomposition.

  It will materialize the matrix ``A`` in memory.

  Args:
    matvec: product between positive definite matrix ``A`` and a vector.
    b: array.

  Returns:
    array with same structure as ``b``.
  """
  if len(b.shape) == 0:
    return b / _materialize_array(matvec, b.shape)
  elif len(b.shape) == 1:
    A = _materialize_array(matvec, b.shape)
    return jax.scipy.linalg.solve(A, b, sym_pos=True)
  elif len(b.shape) == 2:
    A = _materialize_array(matvec, b.shape)
    return  jax.scipy.linalg.solve(A, b.ravel(), sym_pos=True).reshape(*b.shape)
  else:
    raise NotImplementedError


def solve_cg(matvec: Callable,
             b: Any,
             ridge: Optional[float] = None,
             init: Optional[Any] = None,
             **kwargs) -> Any:
  """Solves ``A x = b`` using conjugate gradient.

  It assumes that ``A`` is  a Hermitian, positive definite matrix.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by conjugate gradient.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  if ridge is not None:
    matvec = _make_ridge_matvec(matvec, ridge=ridge)
  return jax.scipy.sparse.linalg.cg(matvec, b, x0=init, **kwargs)[0]


def _rmatvec(matvec, x):
  """Computes A^T x, from matvec(x) = A x, where A is square."""
  transpose = jax.linear_transpose(matvec, x)
  return transpose(x)[0]


def _normal_matvec(matvec, x):
  """Computes A^T A x from matvec(x) = A x."""
  matvec_x, vjp = jax.vjp(matvec, x)
  return vjp(matvec_x)[0]


def solve_normal_cg(matvec: Callable,
                    b: Any,
                    ridge: Optional[float] = None,
                    init: Optional[Any] = None,
                    **kwargs) -> Any:
  """Solves the normal equation ``A^T A x = A^T b`` using conjugate gradient.

  This can be used to solve Ax=b using conjugate gradient when A is not
  hermitian, positive definite.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by normal conjugate gradient.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  def _matvec(x):
    """Computes A^T A x."""
    return _normal_matvec(matvec, x)

  if ridge is not None:
    _matvec = _make_ridge_matvec(_matvec, ridge=ridge)

  Ab = _rmatvec(matvec, b)

  return jax.scipy.sparse.linalg.cg(_matvec, Ab, x0=init, **kwargs)[0]


def solve_gmres(matvec: Callable,
                b: Any,
                ridge: Optional[float] = None,
                init: Optional[Any] = None,
                tol: float = 1e-5,
                **kwargs) -> Any:
  """Solves ``A x = b`` using gmres.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by gmres.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  if ridge is not None:
    matvec = _make_ridge_matvec(matvec, ridge=ridge)
  return jax.scipy.sparse.linalg.gmres(matvec, b, tol=tol, x0=init, **kwargs)[0]


def solve_bicgstab(matvec: Callable,
                   b: Any,
                   ridge: Optional[float] = None,
                   init: Optional[Any] = None,
                   **kwargs) -> Any:
  """Solves ``A x = b`` using bicgstab.

  Args:
    matvec: product between ``A`` and a vector.
    b: pytree.
    ridge: optional ridge regularization.
    init: optional initialization to be used by bicgstab.
    **kwargs: additional keyword arguments for solver.

  Returns:
    pytree with same structure as ``b``.
  """
  if ridge is not None:
    matvec = _make_ridge_matvec(matvec, ridge=ridge)
  return jax.scipy.sparse.linalg.bicgstab(matvec, b, x0=init, **kwargs)[0]
