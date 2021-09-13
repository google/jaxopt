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

"""Implicit differentiation of the lasso based on a sparse implementation."""

import time
from absl import app
import jax
import jax.numpy as jnp
import numpy as onp
from jaxopt import implicit_diff
from jaxopt import linear_solve
from jaxopt import OptaxSolver
from jaxopt import prox
from jaxopt import objective
from jaxopt._src import test_util
import optax
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

# def main(argv):
#   del argv

# Prepare data.
# X, y = datasets.load_boston(return_X_y=True)

X, y = datasets.make_regression(
  n_samples=30, n_features=10_000, random_state=0)

# X = preprocessing.normalize(X)
# data = (X_tr, X_val, y_tr, y_val)
data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

L = onp.linalg.norm(X, ord=2) ** 2


def optimality_fun(params, lam, data):
  X, y = data
  n_samples = X.shape[0]
  return prox.prox_lasso(
    params - jax.grad(objective.least_squares)(params, (X, y)) * n_samples / L,
    lam * len(y) / L) - params


def make_restricted_optimality_fun(support):
  def restricted_optimality_fun(restricted_params, lam, data):
    # this is suboptimal, I would try to compute restricted_X once for all
    X, y = data
    restricted_X = X[:, support]
    return optimality_fun(restricted_params, lam, (restricted_X, y))
  return restricted_optimality_fun


@implicit_diff.sparse_custom_root(
  optimality_fun=optimality_fun,
  make_restricted_optimality_fun=make_restricted_optimality_fun)
def lasso_solver(init_params, lam, data):
  """Solve Lasso."""
  X_tr, y_tr = data
  # TODO add warm start?
  sol = test_util.lasso_skl(X, y, lam)
  return sol

# @implicit_diff.custom_root(
#   optimality_fun=optimality_fun)
# def lasso_solver(init_params, lam, data):
#   """Solve Lasso."""
#   X_tr, y_tr = data
#   # TODO add warm start?
#   sol = test_util.lasso_skl(X, y, lam)
#   return sol


# Perhaps confusingly, theta is a parameter of the outer objective,
# but l2reg = jnp.exp(theta) is an hyper-parameter of the inner objective.
def outer_objective(theta, init_inner, data):
  """Validation loss."""
  X_tr, X_val, y_tr, y_val = data
  # We use the bijective mapping l2reg = jnp.exp(theta)
  # both to optimize in log-space and to ensure positivity.
  lam = jnp.exp(theta)
  w_fit = lasso_solver(init_inner, lam, (X_tr, y_tr))
  y_pred = jnp.dot(X_val, w_fit)
  loss_value = jnp.mean((y_pred - y_val) ** 2)
  # We return w_fit as auxiliary data.
  # Auxiliary data is stored in the optimizer state (see below).
  return loss_value, w_fit


# Initialize solver.
solver = OptaxSolver(opt=optax.adam(1e-2), fun=outer_objective, has_aux=True)
lam_max = jnp.max(jnp.abs(X.T @ y)) / len(y)
lam = lam_max / 10
theta_init = jnp.log(lam)
theta, state = solver.init(theta_init)
init_w = jnp.zeros(X.shape[1])

t_start = time.time()
# Run outer loop.
for _ in range(10):
  theta, state = solver.update(
    params=theta, state=state, init_inner=init_w, data=data)
  # The auxiliary data returned by the outer loss is stored in the state.
  init_w = state.aux
  print(f"[Step {state.iter_num}] Validation loss: {state.value:.3f}.")
t_ellapsed = time.time() - t_start

# if __name__ == "__main__":
#   app.run(main)
print("Time taken for 10 iterations: %.2f" % t_ellapsed)
