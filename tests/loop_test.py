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
from jaxopt import loop


class LoopTest(jtu.JaxTestCase):

  @parameterized.product(unroll=[True, False], jit=[True, False])
  def test_while_loop(self, unroll, jit):
    def my_pow(x, y):
      def body_fun(val):
        return val * x
      def cond_fun(val):
        return True
      return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=1.0,
                             maxiter=y, unroll=unroll, jit=jit)

    if not unroll and not jit:
      self.assertRaises(ValueError, my_pow, 3, 4)
      return

    self.assertEqual(my_pow(3, 4), pow(3, 4))

    if unroll:
      # unroll=False uses lax.while_loop, whichs is not differentiable.
      self.assertEqual(jax.grad(my_pow)(3.0, 4),
                       jax.grad(jnp.power)(3.0, 4))

  @parameterized.product(unroll=[True, False], jit=[True, False])
  def test_while_loop_stopped(self, unroll, jit):
    def my_pow(x, y, max_val):
      def body_fun(val):
        return val * x
      def cond_fun(val):
        return val < max_val
      return loop.while_loop(cond_fun=cond_fun, body_fun=body_fun, init_val=1.0,
                             maxiter=y, unroll=unroll, jit=jit)

    if not unroll and not jit:
      self.assertRaises(ValueError, my_pow, 3, 4, max_val=81)
      return

    # We asked for pow(3, 6) but due to max_val, we get pow(3, 4).
    self.assertEqual(my_pow(3, 6, max_val=81), pow(3, 4))

    if unroll:
      self.assertEqual(jax.grad(my_pow)(3.0, 6, max_val=81),
                       jax.grad(jnp.power)(3.0, 4))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
