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
Binary kernel SVM with intercept.
=================================

The dual objective of binary kernel SVMs with an intercept contains both
box constraints and an equality constraint, making it challenging to solve.
The state-of-the-art algorithm to solve this objective is SMO (Sequential
minimal optimization). 

We nevertheless demonstrate in this example how to solve
it by projected gradient descent, by projecting on the constraint set
using projection_box_section.  
  
Since the dual objective is a Quadratic Program we show how to solve
it with BoxOSQP too.
"""

from absl import app
from absl import flags

import jax
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import ProjectedGradient
from jaxopt import BoxOSQP

import numpy as onp
from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm


flags.DEFINE_float("lam", 0.5, "Regularization parameter. Must be positive.")
flags.DEFINE_float("tol", 1e-6, "Tolerance of solvers.")
flags.DEFINE_integer("num_samples", 30, "Size of train set.")
flags.DEFINE_integer("num_features", 5, "Features dimension.")
flags.DEFINE_bool("verbose", False, "Verbosity.")
FLAGS = flags.FLAGS


def binary_kernel_svm_skl(X, y, C):
  print(f"Solve SVM with sklearn.svm.SVC: ")
  K = jnp.dot(X, X.T)
  svc = svm.SVC(kernel="precomputed", C=C, tol=FLAGS.tol).fit(K, y)
  dual_coef = onp.zeros(K.shape[0])
  dual_coef[svc.support_] = svc.dual_coef_[0]
  return dual_coef


def binary_kernel_svm_pg(X, y, C):
  print(f"Solve SVM with Projected Gradient: ")

  def objective_fun(beta, X, y):
    """Dual objective of binary kernel SVMs with intercept."""
    # The dual objective is:
    # fun(beta) = 0.5 beta^T K beta - beta^T y
    # subject to
    # sum(beta) = 0
    # 0 <= beta_i <= C if y_i = 1
    # -C <= beta_i <= 0 if y_i = -1
    # where C = 1.0 / lam
    # and K = X X^T
    Kbeta = jnp.dot(X, jnp.dot(X.T, beta))
    return 0.5 * jnp.dot(beta, Kbeta) - jnp.dot(beta, y)

  # Define projection operator.
  w = jnp.ones(X.shape[0])

  def proj(beta, C):
    box_lower = jnp.where(y == 1, 0, -C)
    box_upper = jnp.where(y == 1, C, 0)
    proj_params = (box_lower, box_upper, w, 0.0)
    return projection.projection_box_section(beta, proj_params)

  # Run solver.
  beta_init = jnp.ones(X.shape[0])
  solver = ProjectedGradient(fun=objective_fun,
                             projection=proj,
                             tol=FLAGS.tol, maxiter=500, verbose=FLAGS.verbose)
  beta_fit = solver.run(beta_init, hyperparams_proj=C, X=X, y=y).params

  return beta_fit


def binary_kernel_svm_osqp(X, y, C):
  # The dual objective is:
  # fun(beta) = 0.5 beta^T K beta - beta^T y
  # subject to
  # sum(beta) = 0
  # 0 <= beta_i <= C if y_i = 1
  # -C <= beta_i <= 0 if y_i = -1
  # where C = 1.0 / lam

  print(f"Solve SVM with OSQP: ")

  def matvec_Q(X, beta):
    return jnp.dot(X, jnp.dot(X.T,  beta))
  
  # There qre two types of constraints:
  #   0 <= y_i * beta_i <= C     (1)
  # and:
  #   sum(beta) = 0              (2)
  # The first one involves the identity matrix over the betas.
  # The second one involves their sum (i.e dot product with vector full of 1).
  # We take advantage of matvecs to avoid materializing A in memory.
  # We return a tuple whose entries correspond each type of constraint.
  def matvec_A(_, beta):
    return beta, jnp.sum(beta)
  
  # l, u must have same shape than matvec_A's output.
  l = -jax.nn.relu(-y * C), 0.
  u =  jax.nn.relu( y * C), 0.

  hyper_params = dict(params_obj=(X, -y), params_eq=None, params_ineq=(l, u))
  osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=FLAGS.tol)
  params, _ = osqp.run(init_params=None, **hyper_params)
  beta = params.primal[0]

  return beta


def print_svm_result(beta, threshold=1e-4):
  # Here the vector `beta` of coefficients is signed:
  # its sign depends of the true label of the corresponding example.
  # Hence we use jnp.abs() to detect support vectors.
  is_support_vectors = jnp.abs(beta) > threshold
  print(f"Beta: {beta}")
  print(f"Support vector indices: {onp.where(is_support_vectors)[0]}")
  print("")


def main(argv):
  del argv

  num_samples = FLAGS.num_samples
  num_features = FLAGS.num_features

  # Prepare data.
  X, y = datasets.make_classification(n_samples=num_samples, n_features=num_features,
                                      n_classes=2,
                                      random_state=0)
  X = preprocessing.Normalizer().fit_transform(X)
  y = jnp.array(y * 2. - 1)  # Transform labels from {0, 1} to {-1., 1.}.

  lam = FLAGS.lam
  C = 1./ lam

  # Compare the obtained dual coefficients.
  beta_fit_osqp = binary_kernel_svm_osqp(X, y, C)
  print_svm_result(beta_fit_osqp)

  beta_fit_pg = binary_kernel_svm_pg(X, y, C)
  print_svm_result(beta_fit_pg)

  beta_fit_skl = binary_kernel_svm_skl(X, y, C)
  print_svm_result(beta_fit_skl)
  

if __name__ == "__main__":
  jax.config.update("jax_platform_name", "cpu")
  app.run(main)
