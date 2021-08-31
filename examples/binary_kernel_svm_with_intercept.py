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

"""Binary kernel SVM with intercept.

The dual objective of binary kernel SVMs with an intercept contains both
box constraints and an equality constraint, making it challenging to solve.
The state-of-the-art algorithm to solve this objective is SMO (Sequential
minimal optimization). We nevertheless demonstrate in this example how to solve
it by projected gradient descent, by projecting on the constraint set
using projection_box_section.
"""

from absl import app
import jax.numpy as jnp
from jaxopt import projection
from jaxopt import ProjectedGradient
import numpy as onp
from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm


def objective_fun(beta, lam, K, y):
  """Dual objective of binary kernel SVMs with intercept."""
  # The dual objective is:
  # fun(beta) = 0.5 beta^T K beta - beta^T y
  # subject to
  # sum(beta) = 0
  # 0 <= beta_i <= C if y_i = 1
  # -C <= beta_i <= 0 if y_i = -1
  # where C = 1.0 / lam
  return 0.5 * jnp.dot(beta, jnp.dot(K, beta)) - jnp.dot(beta, y)


def binary_kernel_svm_skl(K, y, lam, tol=1e-5):
  svc = svm.SVC(kernel="precomputed", C=1.0 / lam).fit(K, y)
  dual_coef = onp.zeros(K.shape[0])
  dual_coef[svc.support_] = svc.dual_coef_[0]
  return dual_coef


def main(argv):
  del argv

  # Prepare data.
  X, y = datasets.make_classification(n_samples=20, n_features=5,
                                      n_informative=3, n_classes=2,
                                      random_state=0)
  X = preprocessing.Normalizer().fit_transform(X)
  y = y * 2 - 1  # Transform labels from {0, 1} to {-1, 1}.
  lam = 1.0
  C = 1./ lam
  K = jnp.dot(X, X.T)  # Use a linear kernel.

  # Define projection operator.
  w = jnp.ones(X.shape[0])

  def proj(beta, C):
    box_lower = jnp.where(y == 1, 0, -C)
    box_upper = jnp.where(y == 1, C, 0)
    proj_params = (box_lower, box_upper, w, 0.0)
    return projection.projection_box_section(beta, proj_params)

  # Run solver.
  beta_init = jnp.ones(X.shape[0])
  solver = ProjectedGradient(fun=objective_fun, projection=proj,
                            tol=1e-3, maxiter=500)
  beta_fit = solver.run(beta_init, hyperparams_proj=C, lam=lam, K=K, y=y).params

  # Compare the obtained dual coefficients with sklearn.
  beta_fit_skl = binary_kernel_svm_skl(K, y, lam)
  print(beta_fit)
  print(beta_fit_skl)

if __name__ == "__main__":
  app.run(main)
