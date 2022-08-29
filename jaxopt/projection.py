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

from jaxopt._src.projection import projection_non_negative
from jaxopt._src.projection import projection_box
from jaxopt._src.projection import projection_hypercube
from jaxopt._src.projection import projection_simplex
from jaxopt._src.projection import projection_sparse_simplex
from jaxopt._src.projection import projection_l1_sphere
from jaxopt._src.projection import projection_l1_ball
from jaxopt._src.projection import projection_l2_sphere
from jaxopt._src.projection import projection_l2_ball
from jaxopt._src.projection import projection_linf_ball
from jaxopt._src.projection import projection_hyperplane
from jaxopt._src.projection import projection_halfspace
from jaxopt._src.projection import projection_affine_set
from jaxopt._src.projection import projection_polyhedron
from jaxopt._src.projection import projection_box_section
from jaxopt._src.projection import projection_transport
from jaxopt._src.projection import projection_birkhoff
from jaxopt._src.projection import kl_projection_transport
from jaxopt._src.projection import kl_projection_birkhoff
