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

import time
import jax
import jax.numpy as jnp

from jaxopt import prox
from jaxopt import implicit_diff as idf
from jaxopt._src import test_util
from jaxopt import objective

from sklearn import datasets


def lasso_objective(params, lam, X, y):
  residuals = jnp.dot(X, params) - y
  return 0.5 * jnp.mean(residuals ** 2) / len(y) + lam * jnp.sum(
    jnp.abs(params))


def lasso_solver(params, X, y, lam):
  sol = test_util.lasso_skl(X, y, lam)
  return sol


X, y = datasets.make_regression(
    n_samples=10, n_features=10_000, random_state=0)
lam_max = jnp.max(jnp.abs(X.T @ y)) / len(y)
lam = lam_max / 2
L = jax.numpy.linalg.norm(X, ord=2) ** 2


def make_restricted_optimality_fun(support):
  def restricted_optimality_fun(restricted_params, X, y, lam):
    # this is suboptimal, I would try to compute restricted_X once for all
    restricted_X = X[:, support]
    return lasso_optimality_fun(restricted_params, restricted_X, y, lam)
  return restricted_optimality_fun


def lasso_optimality_fun(params, X, y, lam, tol=1e-4):
  n_samples = X.shape[0]
  return prox.prox_lasso(
    params - jax.grad(objective.least_squares)(params, (X, y)) * n_samples / L, lam * len(y) / L) - params


t_start = time.time()
lasso_solver_decorated = idf.custom_root(lasso_optimality_fun)(lasso_solver)
sol = test_util.lasso_skl(X=X, y=y, lam=lam)
J = jax.jacrev(lasso_solver_decorated, argnums=3)(None, X, y, lam)
t_custom = time.time() - t_start


t_start = time.time()
lasso_solver_decorated = idf.sparse_custom_root(
    lasso_optimality_fun, make_restricted_optimality_fun)(lasso_solver)
sol = test_util.lasso_skl(X=X, y=y, lam=lam)
J = jax.jacrev(lasso_solver_decorated, argnums=3)(None, X, y, lam)
t_custom_sparse = time.time() - t_start


print("Time taken to compute the Jacobian %.3f" % t_custom)
print("Time taken to compute the Jacobian with the sparse implementation %.3f" % t_custom_sparse)
