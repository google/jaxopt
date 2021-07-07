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

"""Non-negative matrix factorizaton (NMF) using alternating minimization."""

from absl import app
from absl import flags

import jax
import jax.numpy as jnp

from jaxopt import block_cd2 as block_cd
from jaxopt import prox

from jaxopt.objectives import least_squares_objective

import numpy as onp

from sklearn import datasets


flags.DEFINE_string("penalty", "l2", "Regularization type.")
flags.DEFINE_float("gamma", 1.0, "Regularization strength.")
FLAGS = flags.FLAGS


def nnreg(U, V_init, X, maxiter=150):
  """Regularized non-negative regression.

  We solve::

  min_{V >= 0} mean((U V^T - X) ** 2) + 0.5 * gamma * ||V||^2_2

  or

  min_{V >= 0} mean((U V^T - X) ** 2) +  gamma * ||V||_1
  """
  if FLAGS.penalty == "l2":
    block_prox = prox.prox_non_negative_ridge
  elif FLAGS.penalty == "l1":
    block_prox = prox.prox_non_negative_lasso
  else:
    raise ValueError("Invalid penalty.")

  bcd = block_cd.BlockCoordinateDescent(fun=least_squares_objective,
                                        block_prox=block_prox,
                                        maxiter=maxiter)
  sol = bcd.run(init_params=V_init.T, hyperparams_prox=FLAGS.gamma, data=(U, X))
  return sol.params.T  # approximate solution V


def reconstruction_error(U, V, X):
  """Computes (unregularized) reconstruction error."""
  UV = jnp.dot(U, V.T)
  return 0.5 * jnp.mean((UV - X) ** 2)


def nmf(U_init, V_init, X, maxiter=10):
  """NMF by alternating minimization.

  We solve

    min_{U >= 0, V>= 0} ||U V^T - X||^2 + 0.5 * gamma * (||U||^2_2 + ||V||^2_2)

  or

    min_{U >= 0, V>= 0} ||U V^T - X||^2 + gamma * (||U||_1 + ||V||_1)
  """
  U, V = U_init, V_init

  error = reconstruction_error(U, V, X)
  print(f"STEP: 0; Error: {error:.3f}")
  print()

  for step in range(1, maxiter + 1):
    print(f"STEP: {step}")

    V = nnreg(U, V, X, maxiter=150)
    error = reconstruction_error(U, V, X)
    print(f"Error: {error:.3f} (V update)")

    U = nnreg(V, U, X.T, maxiter=150)
    error = reconstruction_error(U, V, X)
    print(f"Error: {error:.3f} (U update)")
    print()


def main(argv):
  del argv

  # Prepare data.
  X, _ = datasets.load_boston(return_X_y=True)
  X = jnp.sqrt(X ** 2)

  n_samples = X.shape[0]
  n_features = X.shape[1]
  n_components = 10

  rng = onp.random.RandomState(0)
  U = jnp.array(rng.rand(n_samples, n_components))
  V = jnp.array(rng.rand(n_features, n_components))

  # Run the algorithm.
  print("penalty:", FLAGS.penalty)
  print("gamma", FLAGS.gamma)
  print()

  nmf(U, V, X, maxiter=30)

if __name__ == "__main__":
  app.run(main)
