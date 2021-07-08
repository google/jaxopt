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

"""Multiclass linear SVM (without intercept)."""

from absl import app
import jax.numpy as jnp
from jaxopt import BlockCoordinateDescent
from jaxopt import objectives
from jaxopt import projection
from jaxopt import prox
from sklearn import datasets
from sklearn import preprocessing
from sklearn import svm


def multiclass_linear_svm_skl(X, y, lam, tol=1e-5):
  svc = svm.LinearSVC(loss="hinge", dual=True, multi_class="crammer_singer",
                      C=1.0 / lam, fit_intercept=False, tol=tol).fit(X, y)
  return svc.coef_.T


def main(argv):
  del argv

  # Generate data.
  n_samples, n_classes = 20, 3
  X, y = datasets.make_classification(n_samples=n_samples, n_features=5,
                                      n_informative=3, n_classes=n_classes,
                                      random_state=0)
  Y = preprocessing.LabelBinarizer().fit_transform(y)
  Y = jnp.array(Y)

  # Set up parameters.
  block_prox = prox.make_prox_from_projection(projection.projection_simplex)
  fun = objectives.multiclass_linear_svm_dual
  data = (X, Y)
  lam = 1000.0
  beta_init = jnp.ones((n_samples, n_classes)) / n_classes

  # Run solver.
  bcd = BlockCoordinateDescent(fun=fun, block_prox=block_prox,
                               maxiter=3500, tol=1e-5)
  sol = bcd.run(beta_init, hyperparams_prox=None, lam=lam, data=data)

  # Compare against sklearn.
  W_skl = multiclass_linear_svm_skl(X, y, lam)
  W_fit = jnp.dot(X.T, (Y - sol.params)) / lam

  print(W_skl)
  print(W_fit)


if __name__ == "__main__":
  app.run(main)
