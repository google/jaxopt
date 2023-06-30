# Copyright 2023 Google LLC
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
import numpy as onp

from jaxopt._src.cond import cond
from jaxopt._src import test_util


class CondTest(test_util.JaxoptTestCase):

  @parameterized.product(jit=[False, True])
  def test_cond(self, jit):
    def true_fun(x):
      return x
    def false_fun(x):
      return jnp.zeros_like(x)
        
    def my_relu(x):
      return cond(jnp.sum(x)>0, true_fun, false_fun, x, jit=jit)

    if jit:
      x = onp.array([1.])
    else:
      x = jnp.array([1.])
    self.assertEqual(jax.nn.relu(x), my_relu(x))

if __name__ == '__main__':
  absltest.main()
