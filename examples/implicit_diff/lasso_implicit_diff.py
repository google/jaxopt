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

from absl import app
from absl import flags

import jax
import jax.numpy as jnp

from jaxopt import BlockCoordinateDescent
from jaxopt import objective
from jaxopt import OptaxSolver
from jaxopt import prox
from jaxopt import ProximalGradient
import optax

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

flags.DEFINE_bool("unrolling", False, "Whether to use unrolling.")
flags.DEFINE_string("solver", "bcd", "Solver to use (bcd or pg).")
FLAGS = flags.FLAGS


def outer_objective(theta, init_inner, data):
  """Validation loss."""
  X_tr, X_val, y_tr, y_val = data
  # We use the bijective mapping lam = jnp.exp(theta) to ensure positivity.
  lam = jnp.exp(theta)

  if FLAGS.solver == "pg":
    solver = ProximalGradient(
        fun=objective.least_squares,
        prox=prox.prox_lasso,
        implicit_diff=not FLAGS.unrolling,
        maxiter=500)
  elif FLAGS.solver == "bcd":
    solver = BlockCoordinateDescent(
        fun=objective.least_squares,
        block_prox=prox.prox_lasso,
        implicit_diff=not FLAGS.unrolling,
        maxiter=500)
  else:
    raise ValueError("Unknown solver.")

  # The format is run(init_params, hyperparams_prox, *args, **kwargs)
  # where *args and **kwargs are passed to `fun`.
  w_fit = solver.run(init_inner, lam, (X_tr, y_tr)).params

  y_pred = jnp.dot(X_val, w_fit)
  loss_value = jnp.mean((y_pred - y_val) ** 2)

  # We return w_fit as auxiliary data.
  # Auxiliary data is stored in the optimizer state (see below).
  return loss_value, w_fit


def main(argv):
  del argv

  print("Solver:", FLAGS.solver)
  print("Unrolling:", FLAGS.unrolling)

  # Prepare data.
  X, y = datasets.load_boston(return_X_y=True)
  X = preprocessing.normalize(X)
  # data = (X_tr, X_val, y_tr, y_val)
  data = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

  # Initialize solver.
  solver = OptaxSolver(opt=optax.adam(1e-2), fun=outer_objective, has_aux=True)
  theta_init = 1.0
  theta, state = solver.init(theta_init)
  init_w = jnp.zeros(X.shape[1])

  # Run outer loop.
  for _ in range(10):
    theta, state = solver.update(params=theta, state=state, init_inner=init_w,
                                 data=data)
    # The auxiliary data returned by the outer loss is stored in the state.
    init_w = state.aux
    print(f"[Step {state.iter_num}] Validation loss: {state.value:.3f}.")

if __name__ == "__main__":
  app.run(main)
