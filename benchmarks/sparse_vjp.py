import time
import jax

import jax.numpy as jnp
import numpy as onp

from jaxopt import prox
from jaxopt import implicit_diff as idf
from jaxopt._src import test_util
from jaxopt import linear_solve
from jaxopt import objective

from sklearn import datasets

X, y = datasets.make_regression(
  n_samples=100, n_features=100_000, random_state=0)

L = onp.linalg.norm(X, ord=2) ** 2


def optimality_fun(params, X, y, lam):
  n_samples = X.shape[0]
  return prox.prox_lasso(
    params - jax.grad(objective.least_squares)(params, (X, y)) * n_samples / L,
    lam * len(y) / L) - params


def make_restricted_optimality_fun(support):
  def restricted_optimality_fun(restricted_params, X, y, lam):
    # this is suboptimal, I would try to compute restricted_X once for all
    restricted_X = X[:, support]
    return optimality_fun(restricted_params, restricted_X, y, lam)
  return restricted_optimality_fun


lam_max = jnp.max(jnp.abs(X.T @ y)) / len(y)
lam = lam_max / 2
t_start = time.time()
sol = test_util.lasso_skl(X, y, lam)
t_optim = time.time() - t_start

onp.random.seed(0)
rand = onp.random.normal(0, 1, len(sol))
dict_times = {}
dict_grad = {}

for maxiter in [10, 100, 1000, 2000]:
  def solve(matvec, b):
    return linear_solve.solve_normal_cg(
      matvec, b, None, tol=1e-32, maxiter=maxiter)

  vjp = lambda g: idf.root_vjp(
    optimality_fun=optimality_fun,
    sol=sol,
    args=(X, y, lam),
    cotangent=g,
    solve=solve)[2]  # vjp w.r.t. lam

  t_start = time.time()
  grad = vjp(rand)
  t_jac = time.time() - t_start
  dict_times[maxiter] = t_jac
  dict_grad[maxiter] = grad.copy()


def solve_sparse(matvec, b):
  return linear_solve.solve_cg(
    matvec, b, None, tol=1e-32, maxiter=(sol != 0).sum())


vjp_sparse = lambda g: idf.sparse_root_vjp(
  optimality_fun=optimality_fun,
  make_restricted_optimality_fun=make_restricted_optimality_fun,
  sol=sol,
  args=(X, y, lam),
  cotangent=g,
  solve=solve_sparse)[2]  # vjp w.r.t. lam

vjp_sparse2 = lambda g: idf.sparse_root_vjp2(
  optimality_fun=optimality_fun,
  sol=sol,
  args=(X, y, lam),
  cotangent=g,
  solve=solve_sparse)[2]  # vjp w.r.t. lam

t_start = time.time()
grad_sparse = vjp_sparse(rand)
t_jac_sparse = time.time() - t_start

t_start = time.time()
grad_sparse2 = vjp_sparse(rand)
t_jac_sparse2 = time.time() - t_start

print("Time taken to solve the Lasso optimization problem %.3f" % t_optim)
for maxiter in dict_times.keys():
  print("Time taken to compute the gradient with n= %i iterations %.3f | distance to the sparse gradient %.e" % (
    maxiter, dict_times[maxiter], jnp.linalg.norm(dict_grad[maxiter] - grad_sparse) / grad_sparse))
print("Time taken to compute the gradient with the sparse implementation %.3f" % t_jac_sparse)
print("Time taken to compute the gradient with the sparse2 implementation %.3f" % t_jac_sparse2)


# Computation time are the same, which is very weird to me
# However, the Jacobian computed the sparse way is much closer to the real
# Jacobian
