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

import unittest
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from jaxopt import LevenbergMarquardt
from jaxopt._src import linear_solve
from jaxopt._src import test_util

import numpy as onp


def _rosenbrock(x):
  return jnp.sqrt(2) * jnp.array([10 * (x[1] - x[0]**2), (1 - x[0])])


def _scaled_meyer_residual(coeffs, x, y):
  return 1e-3 * y - coeffs[0] * jnp.exp(10 * coeffs[1] / (x + coeffs[2]) - 13)


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
    res1 = res1.at[:, i].set(lm.run(coeffs, x + eps_at(i), y).params)
    res2 = res2.at[:, i].set(lm.run(coeffs, x - eps_at(i), y).params)

  twoeps = 2 * eps
  return (res1 - res2) / twoeps


def _custom_solve_inv(matvec, b, ridge, init=None):
  matvec = linear_solve._make_ridge_matvec(matvec, ridge=ridge)
  A = linear_solve._materialize_array(matvec, b.shape)
  return jnp.dot(jnp.linalg.inv(A), b)

# Uncomment this line to test in x64 
# Note: this line needs to be defined before the Test Class
# jax.config.update('jax_enable_x64', True)

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

    # Meyers fit data from example 6.21 of K. Madsen & H. B. Nielsen in the 
    # book "Introduction to Optimization and Data Fitting".
    self.scaled_meyer_x = onp.linspace(0.5, 1.25, 16)
    self.scaled_meyer_y = onp.array([
        34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744, 8261, 7030,
        6005, 5147, 4427, 3820, 3307, 2872
    ])

  @unittest.skipIf(jax.config.jax_enable_x64, 'test requires X32')
  def test_rosenbrock_x32(self):
    lm = LevenbergMarquardt(
        residual_fun=_rosenbrock,
        damping_parameter=1e-3,
        maxiter=16)
    optimize_info = lm.run(onp.array([-1.2, 1]))

    self.assertArraysAllClose(optimize_info.params,
                              onp.array([0.999948, 0.999894], float))
    self.assertAllClose(optimize_info.state.iter_num, 14)

  @unittest.skipIf(jax.config.jax_enable_x64, 'test requires X32')
  @parameterized.product(materialize_jac=[True, False])
  def test_rosenbrock_x32_materialize_jac(self, materialize_jac):

    lm = LevenbergMarquardt(
        residual_fun=_rosenbrock,
        damping_parameter=1e-3,
        maxiter=16,
        materialize_jac=materialize_jac)
    optimize_info = lm.run(onp.array([-1.2, 1]))

    self.assertArraysAllClose(optimize_info.params,
                              onp.array([0.999948, 0.999894], float))
    self.assertAllClose(optimize_info.state.iter_num, 14)

  # This unit test is for checking rosenbrock function based on the example
  # provided by K. Madsen & H. B. Nielsen in the book "Introduction to
  # Optimization and Data Fitting" Example 6.9. page 123.
  @unittest.skipIf(not jax.config.jax_enable_x64, 'test requires X64')
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

  @unittest.skipIf(not jax.config.jax_enable_x64, 'test requires X64')
  @parameterized.product(materialize_jac=[True, False])
  def test_rosenbrock_x64_materialize_jac(self, materialize_jac):
    lm = LevenbergMarquardt(
        residual_fun=_rosenbrock,
        damping_parameter=1e-3,
        stop_criterion='madsen-nielsen',
        xtol=1e-12,
        gtol=1e-8,
        maxiter=16,
        materialize_jac=materialize_jac)
    optimize_info = lm.run(onp.array([-1.2, 1]))

    self.assertArraysAllClose(
        optimize_info.params - onp.array([1, 1]),
        onp.array([-4.07e-09, -8.16e-09]),
        atol=1e-11)
    self.assertAllClose(optimize_info.state.iter_num, 16)

  @unittest.skipIf(jax.config.jax_enable_x64, 'test requires X32')
  @parameterized.product(materialize_jac=[True, False], 
                         solver=['cholesky', 'lu', 'qr', 'inv'])
  def test_scaled_meyer_x32(self, materialize_jac, solver):
    if materialize_jac == False and solver == 'qr':
      self.skipTest("QR factorization solver requires materialization of Jacobian.")
    
    lm = LevenbergMarquardt(
        residual_fun=_scaled_meyer_residual,
        damping_parameter=10,
        tol=1e-3,
        maxiter=200,
        solver=solver,
        materialize_jac=materialize_jac,
        geodesic=True)
    optimize_info = lm.run(
        onp.asarray([40, 50, 30], dtype=float),
        self.scaled_meyer_x,
        self.scaled_meyer_y)

    self.assertArraysAllClose(
        optimize_info.params, onp.array([2.48, 6.18, 3.45], dtype=float), atol=5e-2, rtol=2e-2)
    
  @unittest.skipIf(not jax.config.jax_enable_x64, 'test requires X64')
  @parameterized.product(materialize_jac=[True, False], 
                         solver=['cholesky', 'lu', 'qr', 'inv'])
  def test_scaled_meyer_x64(self, materialize_jac, solver):
    if materialize_jac == False and solver == 'qr':
      self.skipTest("QR factorization solver requires materialization of Jacobian.")
    
    lm = LevenbergMarquardt(
        residual_fun=_scaled_meyer_residual,
        damping_parameter=10,
        tol=1e-3,
        maxiter=200,
        solver=solver,
        materialize_jac=materialize_jac,
        geodesic=True)
    optimize_info = lm.run(
        onp.asarray([40, 50, 30], dtype=float),
        self.scaled_meyer_x,
        self.scaled_meyer_y)

    self.assertArraysAllClose(
        optimize_info.params, onp.array([2.48, 6.18, 3.45], dtype=float), rtol=5*1e-3)

  # This unit test is for checking rosenbrock function based on the example
  # provided by K. Madsen & H. B. Nielsen in the book "Introduction to
  # Optimization and Data Fitting" Example 6.10-. page 124. Data taken from
  # immoptibox "A MATLAB TOOLBOX FOR OPTIMIZATION AND DATA FITTING" H.B. Nielsen
  @unittest.skipIf(not jax.config.jax_enable_x64, 'test requires X64')
  @parameterized.product(materialize_jac=[True, False],
                         solver=['cholesky', 'lu', 'qr', 'inv', _custom_solve_inv],
                         geodesic=[True, False],)
  def test_double_exponential_fit_x64(self, materialize_jac, solver, geodesic):
    if materialize_jac == False and solver == 'qr':
      self.skipTest("QR factorization solver requires materialization of Jacobian.")
    lm = LevenbergMarquardt(
        residual_fun=_double_exponential_residual,
        damping_parameter=1e-3,
        stop_criterion='madsen-nielsen',
        xtol=1e-14,
        gtol=1e-8,
        maxiter=100,
        solver=solver,
        geodesic=geodesic,
        materialize_jac=materialize_jac)
    optimize_info = lm.run(
        onp.asarray([1, -1, 1, 2], dtype=float),
        self.double_exponential_x,
        self.double_exponential_y)

    # Check the final result up to 4 sig digits.
    self.assertArraysAllClose(
        optimize_info.params, onp.array([4, -4, 4, 5], dtype=float), atol=5*1e-3)

  # TODO(vroulet) This test is failing on github, producing NaNs. Needs to fix algo.
  # @parameterized.product(materialize_jac=[True, False],
  #                        solver=['cholesky', 'lu', 'qr', 'inv', _custom_solve_inv],
  #                        geodesic=[True, False],)
  # def test_double_exponential_fit_x32(self, materialize_jac, solver, geodesic):
  #   if materialize_jac == False and solver == 'qr':
  #     self.skipTest("QR factorization solver requires materialization of Jacobian.")
  #   lm = LevenbergMarquardt(
  #       residual_fun=_double_exponential_residual,
  #       damping_parameter=1e-3,
  #       stop_criterion='madsen-nielsen',
  #       xtol=1e-6,
  #       gtol=1e-6,
  #       maxiter=200,
  #       solver=solver,
  #       geodesic=geodesic,
  #       materialize_jac=materialize_jac)
  #   optimize_info = lm.run(
  #       onp.asarray([1, -1, 1, 2], dtype=float),
  #       self.double_exponential_x,
  #       self.double_exponential_y)

  #   # Check the final result up to 4 sig digits.
  #   self.assertArraysAllClose(
  #       optimize_info.params, onp.array([4, -4, 4, 5], dtype=float), atol=0.5)

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

  def test_has_aux_false(self):
    lm = LevenbergMarquardt(
        residual_fun=lambda x, y: jnp.abs(x - y), tol=1e-6, has_aux=False)
    x_init = jnp.zeros((10,))
    x_gt = jnp.linspace(0, 5, 10)
    x_opt, state = lm.run(x_init, x_gt)

    self.assertAllClose(x_opt, x_gt, atol=1e-6)
    self.assertIsNone(state.aux)

  def test_has_aux_true(self):
    lm = LevenbergMarquardt(
        residual_fun=lambda x, y: (jnp.abs(x - y), x**2),
        tol=1e-6,
        has_aux=True)
    x_init = jnp.zeros((10,))
    x_gt = jnp.linspace(0, 5, 10)
    x_opt, state = lm.run(x_init, x_gt)

    self.assertAllClose(x_opt, x_gt, atol=1e-6)
    self.assertAllClose(state.aux, x_gt**2, atol=1e-6)

  def test_scalar_output_fun(self):
    lm = LevenbergMarquardt(
        residual_fun=lambda x: x @ x,
        tol=1e-1,)
    x_init = jnp.ones((2,))
    x_opt, _ = lm.run(x_init)

    self.assertAllClose(x_opt, jnp.zeros((2,)), atol=1e0)

if __name__ == '__main__':
  absltest.main()
