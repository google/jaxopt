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

from absl.testing import absltest
from absl.testing import parameterized
from typing import Optional
import unittest
from unittest import SkipTest

import jax
import jax.numpy as jnp

from jaxopt._src import linear_solve
from jaxopt._src import test_util
from jaxopt import LevenbergMarquardt

import numpy as onp


def _rosenbrock(x):
  return jnp.sqrt(2) * jnp.array([10 * (x[1] - x[0]**2), (1 - x[0])])


def _double_exponential(coeffs, x):
  return coeffs[0] * jnp.exp(-coeffs[2] * x) + coeffs[1] * jnp.exp(
      -coeffs[3] * x)


def _double_exponential_residual(coeffs, x, y):
  return y - _double_exponential(coeffs, x)


def _enzyme_reaction_residual_model(coeffs, x, y):
  return y - coeffs[0] * x / (coeffs[1] + x)


def _enzyme_reaction_residual_model_jac(coeffs, x, y, eps=1e-5):
  """Return the numerical Jacobian."""
  lm = LevenbergMarquardt(
      residual_fun=_enzyme_reaction_residual_model,)

  # Sets eps only at idx, the rest is zero
  eps_at = lambda idx: onp.array([int(i == idx)*eps for i in range(len(x))])

  res1 = jnp.zeros((len(coeffs), len(x)))
  res2 = jnp.zeros((len(coeffs), len(x)))
  for i in range(len(x)):
    res1 = res1.at[:,i].set(lm.run(coeffs, x + eps_at(i), y).params)
    res2 = res2.at[:,i].set(lm.run(coeffs, x - eps_at(i), y).params)

  twoeps = 2 * eps
  return (res1 - res2) / twoeps


def _custom_solve_inv(matvec, b, ridge, init=None):
  matvec = linear_solve._make_ridge_matvec(matvec, ridge=ridge)
  A = linear_solve._materialize_array(matvec, b.shape)
  return jnp.dot(jnp.linalg.inv(A), b)


class LevenbergMarquardtTest(test_util.JaxoptTestCase):

  def setUp(self):
    super().setUp()

    self.substrate_conc = onp.array(
        [0.038, 0.194, .425, .626, 1.253, 2.500, 3.740])
    self.rate_data = onp.array(
        [0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
    self.init_enzyme_reaction_coeffs = onp.array([0.1, 0.1])

    # Exponential fit data from immoptibox "A MATLAB TOOLBOX FOR OPTIMIZATION
    # AND DATA FITTING" by Hans Bruun Nielsen
    self.double_exponential_x = onp.linspace(0.02, 0.9, 45)
    self.double_exponential_y = onp.array([
        0.090542, 0.124569, 0.179367, 0.195654, 0.269707, 0.286027, 0.289892,
        0.317475, 0.308191, 0.336995, 0.348371, 0.321337, 0.299423, 0.338972,
        0.304763, 0.288903, 0.30082, 0.303974, 0.283987, 0.262078, 0.281593,
        0.267531, 0.218926, 0.225572, 0.200594, 0.197375, 0.18244, 0.183892,
        0.152285, 0.174028, 0.150874, 0.12622, 0.126266, 0.106384, 0.118923,
        0.091868, 0.128926, 0.119273, 0.115997, 0.105831, 0.075261, 0.068387,
        0.090823, 0.085205, 0.067203
    ])

  @unittest.skipIf(jax.config.jax_enable_x64, "test requires X32")
  def test_rosenbrock_x32(self):

    lm = LevenbergMarquardt(
        residual_fun=_rosenbrock,
        damping_parameter=1e-3,
        maxiter=16)
    optimize_info = lm.run(onp.array([-1.2, 1]))

    self.assertArraysAllClose(optimize_info.params,
                              onp.array([0.999948, 0.999894], float))
    self.assertAllClose(optimize_info.state.iter_num, 14)

  # This unit test is for checking rosenbrock function based on the example
  # provided by K. Madsen & H. B. Nielsen in the book "Introduction to
  # Optimization and Data Fitting" Example 6.9. page 123.
  @unittest.skipIf(not jax.config.jax_enable_x64, "test requires X64")
  def test_rosenbrock_x64(self):

    lm = LevenbergMarquardt(
        residual_fun=_rosenbrock,
        damping_parameter=1e-3,
        stop_criterion='madsen-nielsen',
        xtol=1e-12,
        gtol=1e-8,
        maxiter=16)
    optimize_info = lm.run(onp.array([-1.2, 1]))

    self.assertArraysAllClose(
        optimize_info.params - onp.array([1, 1]),
        onp.array([-4.07e-09, -8.16e-09]),
        atol=1e-11)
    self.assertAllClose(optimize_info.state.iter_num, 16)

  @unittest.skipIf(jax.config.jax_enable_x64, "test requires X32")
  @parameterized.product(geodesic=[True, False])
  def test_double_exponential_fit_x32(self, geodesic):
    lm = LevenbergMarquardt(
        residual_fun=_double_exponential_residual,
        damping_parameter=1e-3,
        tol=1e-5,
        maxiter=200,
        solver='cholesky',
        geodesic=geodesic)
    optimize_info = lm.run(
        onp.asarray([2, -2, 2, 3], dtype=float),
        self.double_exponential_x,
        self.double_exponential_y)

    # Check the final result up to 4 sig digits.
    self.assertArraysAllClose(
        optimize_info.params, onp.array([4, -4, 4, 5], dtype=float), atol=0.5)

  # This unit test is for checking rosenbrock function based on the example
  # provided by K. Madsen & H. B. Nielsen in the book "Introduction to
  # Optimization and Data Fitting" Example 6.10-. page 124. Data taken from
  # immoptibox "A MATLAB TOOLBOX FOR OPTIMIZATION AND DATA FITTING" H.B. Nielsen
  @unittest.skipIf(not jax.config.jax_enable_x64, "test requires X64")
  @parameterized.product(solver=['cholesky', 'inv', _custom_solve_inv],
                         geodesic=[True, False])
  def test_double_exponential_fit_x64(self, solver, geodesic):
    lm = LevenbergMarquardt(
        residual_fun=_double_exponential_residual,
        damping_parameter=1e-3,
        stop_criterion='madsen-nielsen',
        xtol=1e-14,
        gtol=1e-8,
        maxiter=100,
        solver=solver,
        geodesic=geodesic)
    optimize_info = lm.run(
        onp.asarray([1, -1, 1, 2], dtype=float),
        self.double_exponential_x,
        self.double_exponential_y)

    # Check the final result up to 4 sig digits.
    self.assertArraysAllClose(
        optimize_info.params, onp.array([4, -4, 4, 5], dtype=float), atol=1e-3)
    if geodesic:
      self.assertAllClose(optimize_info.state.iter_num, 20)
    else:
      self.assertAllClose(optimize_info.state.iter_num, 62)

  @parameterized.product(implicit_diff=[True, False])
  def test_enzyme_reaction_implicit_diff(self, implicit_diff):
    jac_num = _enzyme_reaction_residual_model_jac(
        self.init_enzyme_reaction_coeffs, self.substrate_conc, self.rate_data)

    lm = LevenbergMarquardt(
        residual_fun=_enzyme_reaction_residual_model,
        tol=1e-6,
        implicit_diff=implicit_diff)

    def wrapper(substrate_conc):
      return lm.run(
          self.init_enzyme_reaction_coeffs,
          substrate_conc,
          self.rate_data).params
    jac_custom = jax.jacrev(wrapper)(self.substrate_conc)

    self.assertArraysAllClose(jac_num, jac_custom, atol=5e-2)

if __name__ == '__main__':
  absltest.main()
