import time
import jax

import jax.numpy as jnp

from jaxopt import prox
from jaxopt import implicit_diff as idf
from jaxopt._src import test_util


from sklearn import datasets

X, y = datasets.make_regression(n_samples=10, n_features=10_000, random_state=0)

L = jax.numpy.linalg.norm(X, ord=2) ** 2


def optimality_fun(params, lam, X, y):
  return prox.prox_lasso(
    params - X.T @ (X @ params - y) / L, lam * len(y) / L) - params


# def optimality_fun_sparse(params, lam, X, y):
#     support = params != 0
#     res = X[:, support].T @ (X[:, support] @ params[support] - y) / L
#     res = params[support] - res
#     res = prox.prox_lasso(res, lam * len(y) / L)
#     res -= params[support]
#     return res


lam_max = jnp.max(jnp.abs(X.T @ y)) / len(y)
lam = lam_max / 2
t_start = time.time()
sol = test_util.lasso_skl(X, y, lam)
t_optim = time.time() - t_start

vjp = lambda g: idf.root_vjp(optimality_fun=optimality_fun,
                             sol=sol,
                             args=(lam, X, y),
                             cotangent=g)[0]  # vjp w.r.t. lam

vjp_sparse = lambda g: idf.sparse_root_vjp(
  optimality_fun=optimality_fun,
  sol=sol,
  args=(lam, X, y),
  cotangent=g)[0]  # vjp w.r.t. lam

t_start = time.time()
I = jnp.eye(len(sol))
J = jax.vmap(vjp)(I)
t_jac = time.time() - t_start

t_start = time.time()
I = jnp.eye(len(sol))
J_sparse = jax.vmap(vjp_sparse)(I)
t_jac_sparse = time.time() - t_start

print("Time taken to solve the Lasso optimization problem %.3f" % t_optim)
print("Time taken to compute the Jacobian %.3f" % t_jac)
print("Time taken to compute the Jacobian with the sparse implementation %.3f" % t_jac_sparse)


# Computation time are the same, which is very weird to me
# However, the Jacobian computed the sparse way is much closer to the real
# Jacobian
