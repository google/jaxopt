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

from absl.testing import absltest

from jaxopt import support
from jaxopt._src import test_util

import numpy as onp


class SupportTest(test_util.JaxoptTestCase):

  def test_support_all(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    supp = onp.ones_like(x)
    self.assertArraysEqual(support.support_all(x), supp)

  def test_support_nonzero(self):
    rng = onp.random.RandomState(0)
    x = rng.rand(20) * 2 - 1
    x = onp.where(x > 0, x, 0)
    supp = (x != 0).astype(x.dtype)
    self.assertArraysEqual(support.support_nonzero(x), supp)

  def test_support_group_nonzero(self):
    rng = onp.random.RandomState(0)

    x = rng.rand(20) * 2 - 1
    x = onp.where(x > 0, x, 0)
    self.assertFalse(onp.all(x == 0))
    supp = onp.ones_like(x)
    self.assertArraysEqual(support.support_group_nonzero(x), supp)

    x = onp.zeros(20)
    supp = onp.zeros_like(x)
    self.assertArraysEqual(support.support_group_nonzero(x), supp)

if __name__ == '__main__':
  absltest.main()
