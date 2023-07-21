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
import jax.numpy as jnp
from jaxopt._src import test_util
from jaxopt._src.linesearch_util import _init_stepsize
from jaxopt._src.linesearch_util import _setup_linesearch


class LinesearchTest(test_util.JaxoptTestCase):
  @parameterized.product(
      linesearch=["zoom", "backtracking", "hager-zhang"],
      use_gradient=[False, True],
  )
  def test_linesearch_complex_variables(self, linesearch, use_gradient):
    """Test that optimization over complex variable z = x + jy matches equivalent real case"""

    W = jnp.array([[1, -2], [3, 4], [-4 + 2j, 5 - 3j], [-2 - 2j, 6]])

    def C2R(z):
      return jnp.stack((z.real, z.imag)) if z is not None else None

    def R2C(x):
      return x[..., 0, :] + 1j * x[..., 1, :]

    def f(z):
      return W @ z

    def loss_complex(z):
      return jnp.sum(jnp.abs(f(z)) ** 1.5)

    def loss_real(zR):
      return loss_complex(R2C(zR))

    z0 = jnp.array([1 - 1j, 0 + 1j])

    common_args = dict(
        value_and_grad=False,
        has_aux=False,
        maxlsiter=3,
        max_stepsize=1,
        jit=True,
        unroll=False,
        verbose=False,
    )

    ls_R = _setup_linesearch(
        linesearch=linesearch,
        fun=loss_real,
        **common_args,
    )

    ls_C = _setup_linesearch(
        linesearch=linesearch,
        fun=loss_complex,
        **common_args,
    )

    ls_state = _init_stepsize(
        strategy="increase",
        max_stepsize=1e-1,
        min_stepsize=1e-3,
        increase_factor=2.0,
        stepsize=1e-2,
    )

    descent_direction = (
        -jnp.conj(jax.grad(loss_complex)(z0)) if use_gradient else None
    )

    stepsize_R, _ = ls_R.run(
        ls_state, params=C2R(z0), descent_direction=C2R(descent_direction)
    )
    stepsize_C, _ = ls_C.run(
        ls_state, params=z0, descent_direction=descent_direction
    )

    self.assertArraysAllClose(stepsize_R, stepsize_C)


if __name__ == "__main__":
  absltest.main()
