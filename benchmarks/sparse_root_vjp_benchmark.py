# Copyright 2022 Google LLC
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

"""Benchmark VJP of Lasso, with and without specifying the support."""

import time
import numpy as onp
import jax
import jax.numpy as jnp

from typing import Sequence
from absl import app
from sklearn import datasets

from jaxopt import implicit_diff as idf
from jaxopt._src import linear_solve
from jaxopt import objective
from jaxopt import prox
from jaxopt import support
from jaxopt import tree_util
from jaxopt._src import test_util


def lasso_optimality_fun(params, lam, X, y):
  step = params - jax.grad(objective.least_squares)(params, (X, y))
  return prox.prox_lasso(step, l1reg=lam, scaling=1.) - params


def get_vjp(lam, X, y, sol, support_fun, maxiter, solve_fn=linear_solve.solve_cg):
    def solve(matvec, b):
      return solve_fn(matvec, b, tol=1e-6, maxiter=maxiter)

    vjp = lambda g: idf.root_vjp(optimality_fun=lasso_optimality_fun,
                                 support_fun=support_fun,
                                 sol=sol,
                                 args=(lam, X, y),
                                 cotangent=g,
                                 solve=solve)[0]
    return vjp


def benchmark_vjp(vjp, X, supp=None, desc=''):
  t0 = time.time()
  result = jax.vmap(vjp)(jnp.eye(X.shape[1]))
  result.block_until_ready()
  delta_t = time.time() - t0

  if supp is not None:
    result = result[supp]
  print(f'{desc} ({delta_t:.3f} sec.): {result}')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  X, y = datasets.make_regression(n_samples=100, n_features=1000,
                                  n_informative=5, random_state=0)
  print(f'Number of samples: {X.shape[0]:d}')
  print(f'Number of features: {X.shape[1]:d}')

  lam = 1e-3 * onp.max(X.T @ y)
  print(f'Value of lambda: {lam:.5f}')

  sol = test_util.lasso_skl(X, y, lam, tol=1e-6, fit_intercept=False)
  supp = (sol != 0)
  print(f'Size of the support of the solution: {supp.sum():d}')

  optimality = tree_util.tree_l2_norm(lasso_optimality_fun(sol, lam, X, y))
  print(f'Optimality of the solution (L2-norm): {optimality:.5f}')

  jac_num = test_util.lasso_skl_jac(X, y, lam, eps=1e-8)
  print(f'Numerical Jacobian: {jac_num[supp]}')

  # Compute the Jacobian wrt. lambda, without using the information about the
  # support of the solution. This is the default behavior in JAXopt, and
  # requires solving a linear system with a 1000x1000 dense matrix. Ignoring
  # the support of the solution leads to an inacurrate Jacobian.
  vjp = get_vjp(lam, X, y, sol, support.support_all, maxiter=1000)
  benchmark_vjp(vjp, X, supp=supp, desc='Jacobian w/o support, CG')

  vjp = get_vjp(lam, X, y, sol, support.support_all, maxiter=1000,
                solve_fn=linear_solve.solve_normal_cg)
  benchmark_vjp(vjp, X, supp=supp, desc='Jacobian w/o support, normal CG')

  # Compute the Jacobian wrt. lambda, restricting the data X and the solution
  # to the support of the solution. This requires solving a linear system with
  # a 4x4 dense matrix. Restricting the support this way is not jit-friendly,
  # and will require new compilation if the size of the support changes.
  vjp = get_vjp(lam, X[:, supp], y, sol[supp], support.support_all, maxiter=1000)
  benchmark_vjp(vjp, X[:, supp], desc='Jacobian w/ restricted support')

  # Compute the Jacobian wrt. lambda, by masking the linear system to solve
  # with the support of the solution. This requires solving a linear system with
  # a 1000x1000 sparse matrix. Masking with the support is jit-friendly.
  vjp = get_vjp(lam, X, y, sol, support.support_nonzero, maxiter=1000)
  benchmark_vjp(vjp, X, supp=supp, desc='Jacobian w/ masked support')


if __name__ == '__main__':
  app.run(main)
