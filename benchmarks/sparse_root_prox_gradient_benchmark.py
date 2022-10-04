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

"""
Implicit differentiation of lasso.
==================================
"""

import time
from absl import app
from absl import flags

import jax
import jax.numpy as jnp

from jaxopt import objective
from jaxopt import prox
from jaxopt import ProximalGradient
from jaxopt import support
from jaxopt._src import linear_solve

from sklearn import datasets
from sklearn import model_selection


def outer_objective(theta, init_inner, data, support, implicit_diff_solve):
  """Validation loss."""
  X_tr, X_val, y_tr, y_val = data
  # We use the bijective mapping lam = jnp.exp(theta) to ensure positivity.
  lam = jnp.exp(theta)

  solver = ProximalGradient(
    fun=objective.least_squares,
    support=support,
    prox=prox.prox_lasso,
    implicit_diff=True,
    implicit_diff_solve=implicit_diff_solve,
    maxiter=5000,
    tol=1e-10)

  # The format is run(init_params, hyperparams_prox, *args, **kwargs)
  # where *args and **kwargs are passed to `fun`.
  w_fit = solver.run(init_inner, lam, (X_tr, y_tr)).params

  y_pred = jnp.dot(X_val, w_fit)
  loss_value = jnp.mean((y_pred - y_val) ** 2)

  # We return w_fit as auxiliary data.
  # Auxiliary data is stored in the optimizer state (see below).
  return loss_value, w_fit


X, y = datasets.make_regression(
  n_samples=100, n_features=6000, n_informative=10, random_state=0)

n_samples = X.shape[0]
data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

exp_lam_max = jnp.max(jnp.abs(X.T @ y))
lam = jnp.log(exp_lam_max / (100 * n_samples))

init_inner = jnp.zeros(X.shape[1])


t0 = time.time()
grad = jax.grad(outer_objective, has_aux=True)
gradient, coef_ = grad(
  lam, init_inner, data, support.support_all, linear_solve.solve_cg)
gradient.block_until_ready()
delta_t = time.time() - t0
desc='Gradients w/o support, CG'
print(f'{desc} ({delta_t:.3f} sec.): {gradient} ')

t0 = time.time()
grad = jax.grad(outer_objective, has_aux=True)
gradient, coef_ = grad(
  lam, init_inner, data, support.support_all, linear_solve.solve_normal_cg)
gradient.block_until_ready()
delta_t = time.time() - t0
desc='Gradients w/o support, normal CG'
print(f'{desc} ({delta_t:.3f} sec.): {gradient} ')

t0 = time.time()
grad = jax.grad(outer_objective, has_aux=True)
gradient, coef_ = grad(
  lam, init_inner, data, support.support_nonzero, linear_solve.solve_cg)
gradient.block_until_ready()
delta_t = time.time() - t0
desc='Gradients w/ masked support'
print(f'{desc} ({delta_t:.3f} sec.): {gradient}')
