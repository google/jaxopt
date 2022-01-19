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

import jax
from jax import test_util as jtu
import jax.numpy as jnp

from jaxopt import GaussNewton

import numpy as onp


def _enzyme_reaction_residual_model(coeffs, x, y):
  return y - coeffs[0] * x / (coeffs[1] + x)


def _enzyme_reaction_residual_model_jac(coeffs, x, y, eps=1e-5):
  """Return the numerical Jacobian."""
  gn = GaussNewton(
      residual_fun=_enzyme_reaction_residual_model,
      maxiter=100,
      tol=1.0e-6)

  # Sets eps only at idx, the rest is zero
  eps_at = lambda idx: onp.array([int(i == idx)*eps for i in range(len(x))])

  res1 = jnp.zeros((len(coeffs), len(x)))
  res2 = jnp.zeros((len(coeffs), len(x)))
  for i in range(len(x)):
    res1 = res1.at[:,i].set(gn.run(coeffs, x + eps_at(i), y).params)
    res2 = res2.at[:,i].set(gn.run(coeffs, x - eps_at(i), y).params)

  twoeps = 2 * eps
  return (res1 - res2) / twoeps


def _city_temperature_residual_model(coeffs, x, y):
  return y - (coeffs[0] * jnp.sin(x * coeffs[1] + coeffs[2]) + coeffs[3])


class GaussNewtonTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()

    self.substrate_conc = onp.array(
        [0.038, 0.194, .425, .626, 1.253, 2.500, 3.740])
    self.rate_data = onp.array(
        [0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
    self.init_enzyme_reaction_coeffs = onp.array([0.1, 0.1])

    self.months = onp.arange(1, 13)
    self.temperature_record = onp.array([
        61.0, 65.0, 72.0, 78.0, 85.0, 90.0, 92.0, 92.0, 88.0, 81.0, 72.0, 63.0
    ])
    self.init_temperature_record_coeffs = onp.array([10, 0.5, 10.5, 50])

  def test_enzyme_reaction_parameter_fit(self):
    gn = GaussNewton(
        residual_fun=_enzyme_reaction_residual_model,
        maxiter=100,
        tol=1.0e-6)
    optimize_info = gn.run(
        self.init_enzyme_reaction_coeffs,
        self.substrate_conc,
        self.rate_data)

    self.assertArraysAllClose(optimize_info.params,
                              onp.array([0.36183689, 0.55626653]))
    self.assertAllClose(optimize_info.state.iter_num, 6)


  @parameterized.product(implicit_diff=[True, False])
  def test_enzyme_reaction_implicit_diff(self, implicit_diff):
    jac_num = _enzyme_reaction_residual_model_jac(
        self.init_enzyme_reaction_coeffs, self.substrate_conc, self.rate_data)

    gn = GaussNewton(
        residual_fun=_enzyme_reaction_residual_model,
        tol=1.0e-6,
        maxiter=10,
        implicit_diff=implicit_diff)

    def wrapper(substrate_conc):
      return gn.run(
          self.init_enzyme_reaction_coeffs,
          substrate_conc,
          self.rate_data).params
    jac_custom = jax.jacrev(wrapper)(self.substrate_conc)

    self.assertArraysAllClose(jac_num, jac_custom, atol=1e-2)

  def test_temperature_record_four_parameter_fit(self):
    gn = GaussNewton(
        residual_fun=_city_temperature_residual_model,
        tol=1.0e-6)
    optimize_info = gn.run(
        self.init_temperature_record_coeffs,
        self.months,
        self.temperature_record)

    # Checking against the expected values
    self.assertArraysAllClose(
        optimize_info.params,
        onp.array([16.63994555, 0.46327812, 10.85228919, 76.19086103]))
    self.assertAllClose(optimize_info.state.iter_num, 30)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
