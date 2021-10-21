"""Solvers based on Alternating Directions Method of Multipliers (ADMM)."""
# pylint: disable=invalid-name

import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as onp


def factorize(P, A, rho):
  """Returns LU factorization of block matrix [[P + rho*I, A.T], [A, 0]].

  This matrix arises in the linear system associated with minimizing a quadratic
  subject to linear equality constraints.

  Args:
    P: n x n np.array.
    A: m x n: np.array.
    rho: scalar.

  Returns:
    LU factorization as returned by scipy.linalg.lu_factor.
  """
  n = P.shape[0]
  m = A.shape[0]
  Z = jnp.zeros((m, m))
  I = jnp.eye(n)
  L = jsp.linalg.lu_factor(onp.bmat([[P + rho * I, A.T], [A, Z]]))
  return L


@functools.partial(jax.jit, static_argnums=(4, 5))
def qp(L, q, b, x, rho, maxiter):
  """ADMM-based Quadratic Programming.

        minimize_x 0.5*x'P x + q' x subject to A x = b, x >= 0.


  Args:
    L: output of L = factorize(P, A) where P is n x n and A is m x n np arrays.
    q: 1D numpy array of shape (n, ).
    b: 1D numpy array of shape (m, ).
    x: 1D numpy array, initial guess of shape (n, ).
    rho: ADMM penalty parameter.
    maxiter: maximum number of iterations.

  Returns:
    x: solution which is (n, ) 1D np array.
  """
  n = q.shape[0]
  z = jnp.zeros(n)
  u = jnp.zeros(n)

  def body(inputs, none):  # pylint: disable=unused-argument
    x, z, u = inputs
    s = jnp.concatenate((rho * (z - u) - q, b))
    x_nu = jsp.linalg.lu_solve(L, s)
    x = jax.lax.dynamic_slice(x_nu, (0,), (n,))
    z = jnp.maximum(x + u, 0.0)
    u = u + x - z
    return (x, z, u), 0

  return jax.lax.scan(body, (x, z, u), None, length=maxiter)[0]

