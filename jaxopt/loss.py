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

from jaxopt._src.loss import binary_logistic_loss
from jaxopt._src.loss import binary_sparsemax_loss, sparse_plus, sparse_sigmoid
from jaxopt._src.loss import huber_loss
from jaxopt._src.loss import make_fenchel_young_loss
from jaxopt._src.loss import multiclass_logistic_loss
from jaxopt._src.loss import multiclass_sparsemax_loss
from jaxopt._src.loss import binary_hinge_loss
from jaxopt._src.loss import binary_perceptron_loss
from jaxopt._src.loss import multiclass_hinge_loss
from jaxopt._src.loss import multiclass_perceptron_loss
