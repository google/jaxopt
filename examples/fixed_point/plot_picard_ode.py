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

r"""
Anderson acceleration in application to Picard–Lindelöf theorem.
================================================================

Thanks to the `Picard–Lindelöf theorem,
<https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem>`_ we can
reduce differential equation solving to fixed point computations and simple
integration.  More precisely consider the ODE:

.. math::

  y'(t)=f(t,y(t))

of some time-dependant dynamic
:math:`f:\mathbb{R}\times\mathbb{R}^d\rightarrow\mathbb{R}^d` and initial
conditions :math:`y(0)=y_0`.  Then :math:`y` is the fixed point of the following
map:

.. math::

  y(t)=T(y)(t)\mathrel{\mathop:}=y_0+\int_0^t f(s,y(s))\mathrm{d}s

Then we can define the sequence of functions :math:`(\phi_k)` with
:math:`\phi_0=0` recursively as follows:

.. math::

  \phi_{k+1}(t)=T(\phi_k)(t)\mathrel{\mathop:} =
  y_0+\int_0^t f(s,\phi_k(s))\mathrm{d}s

Such sequence converges to the solution of the ODE, i.e.,
:math:`\lim_{k\rightarrow\infty}\phi_k=y`.

In this example we choose :math:`f(t,y(t))=1+y(t)^2`. We know that the
analytical solution is :math:`y(t)=\tan{t}` , which we use as a ground truth to
evaluate our numerical scheme.
We used ``scipy.integrate.cumtrapz`` to perform
integration, but any other integration method can be used.
"""


import jax
import jax.numpy as jnp

from jaxopt import AndersonAcceleration

from jaxopt import objective
from jaxopt.tree_util import tree_scalar_mul, tree_sub

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.pyplot import cm
import scipy.integrate

jax.config.update("jax_platform_name", "cpu")


# Solve the differential equation y'(t)=1+t^2, with solution y(t) = tan(t)
def f(ti, phi):
  return 1 + phi ** 2

def T(phi_cur, ti, y0, dx):
  """Fixed point iteration in the Picard method.
  See: https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem"""
  f_phi = f(ti, phi_cur)
  phi_next = scipy.integrate.cumtrapz(f_phi, initial=y0, dx=dx)
  return phi_next

y0 = 0
num_interpolating_points = 400
t0 = jnp.array(0.)
tmax = 0.95 * (jnp.pi / 2) # stop before pi/2 to ensure convergence
dx = (tmax - t0) / (num_interpolating_points-1)
phi0 = jnp.zeros(num_interpolating_points)
ti = np.linspace(t0, tmax, num_interpolating_points)

sols = [phi0]
aa = AndersonAcceleration(T, history_size=3, maxiter=50, ridge=1e-5, jit=False)
state = aa.init_state(phi0, ti, y0, dx)
sol = phi0
sols.append(sol)
for k in range(aa.maxiter):
  sol, state = aa.update(phi0, state, ti, y0, dx)
  sols.append(sol)
res = sols[-1] - np.tan(ti)
print(f'Error of {jnp.linalg.norm(res)} with ground truth tan(t)')


# vizualize the first 8 iterates to make the figure easier to read
sols = sols[:8]
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1, 1, 1)

colors = cm.plasma(np.linspace(0, 1, len(sols)))
for k, (sol, c) in enumerate(zip(sols, colors)):
  desc = rf'$\phi_{k}$' if k > 0 else rf'$\phi_0=0$'
  ax.plot(ti, sol, '+', c=c, label=desc)
ax.plot(ti, np.tan(ti), '-', c='green', label=r'$y(t)=\tan{(t)}$ (ground truth)')

ax.legend()
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
formula = rf'$\phi_{{k+1}}(t)=\phi_0+\int_0^{{{tmax/2:.2f}\pi}} f(t,\phi_{{k}}(t))\mathrm{{d}}t$'
ax.text(0.42, 0.85, formula, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
fig.suptitle('Anderson acceleration for ODE solving')
plt.show()
