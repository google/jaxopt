Fixed point resolution
======================

This section is concerned with fixed-point resolution, that is finding
:math:`x` such that :math:`T(x, \theta) = x`.

Fixed point iterations
----------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.FixedPointIteration

The `fixed point iteration
<https://en.wikipedia.org/wiki/Fixed-point_iteration>`_ method simply consists
in iterating :math:`x_{n+1}=T(x_n, \theta)`, which is guaranteed to converge to
a fixed point if :math:`x\mapsto T(x,\theta)` is a **contractive map**. See
`Banach fixed-point theorem
<https://en.wikipedia.org/wiki/Banach_fixed-point_theorem>`_ for more details.

Code example::

  from jaxopt import FixedPointIteration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  fpi = FixedPointIteration(fixed_point_fun=T)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)
  print(fpi.run(x_init, theta).params)

``FixedPointIteration`` successfully finds the fixed point ``x = 1``.

Differentiation
~~~~~~~~~~~~~~~

Fixed points can be differentiated with respect to :math:`\theta`::

  from jaxopt import FixedPointIteration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  fpi = FixedPointIteration(fixed_point_fun=T, implicit_diff=True)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)

  def fixed_point(x, theta):
    return fpi.run(x, theta).params

  print(jax.grad(fixed_point, argnums=1)(x_init, theta))  # only gradient
  print(jax.value_and_grad(fixed_point, argnums=1)(x_init, theta))  # both value and gradient

Note that :math:`x(\theta)=2\theta` so :math:`\nabla x(\theta)=2`.

Under the hood, we use the implicit function theorem in order to differentiate
the fixed point.  See the :ref:`implicit differentiation <implicit_diff>`
section for more details.

Anderson acceleration
---------------------

.. autosummary::
  :toctree: _autosummary

    jaxopt.AndersonAcceleration

Anderson acceleration is an iterative method that aims to compute the next
iterate :math:`x_{n}` as a linear combination of the :math:`m` last iterates
:math:`[x_{n-m},x_{n-m+1},\ldots x_{n-1}]`. The coefficients of the
linear combination are computed 'on the fly' at each iteration. As a result,
not only the convergence is faster but the convergence conditions are weakened,
allowing to tackle problems ``FixedPointIteration`` could not.  See `Pollock
and Rebholz <https://arxiv.org/abs/1909.04638>`_ (2020) for more details.

The size of the history :math:`m` (denoted ``history_size`` below) plays a
crucial role in the method's performance. A higher :math:`m` could speed up the
convergence at the cost of higher memory consumption, and more numerical
instabilities.  Those numerical instabilities can be mitigated by increasing
the ``ridge`` regularization hyper-parameter.

Example::

  from jaxopt import AndersonAcceleration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  aa = AndersonAcceleration(fixed_point_fun=T, history_size=5,
                            ridge=1e-6, tol=1e-5)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)
  print(aa.run(x0, theta).params)

For implicit differentiation::

  from jaxopt import AndersonAcceleration

  def T(x, theta):  # contractive map
    return 0.5 * x + theta

  aa = AndersonAcceleration(fixed_point_fun=T, history_size=5,
                            ridge=1e-6, tol=1e-5, implicit_diff=True)
  x_init = jnp.array(0.)
  theta = jnp.array(0.5)

  def fixed_point(x, theta):
    return aa.run(x, theta).params

  print(jax.grad(fixed_point, argnums=1)(x_init, theta))  # only gradient
  print(jax.value_and_grad(fixed_point, argnums=1)(x_init, theta))  # both value and gradient

.. topic:: Examples

  * :ref:`sphx_glr_auto_examples_fixed_point_plot_anderson_accelerate_gd.py`
  * :ref:`sphx_glr_auto_examples_fixed_point_plot_picard_ode.py`

Equivalence with root finding
-----------------------------

Note that if :math:`x` is a fixed point of :math:`T` then :math:`x` is a root
of :math:`F(x, \theta) = T(x, \theta) - x`.  Reciprocally, if :math:`x` is the
root of some :math:`F(x, \theta)` then it is also the fixed point of
:math:`T(x, \theta) = F(x, \theta) + x`.  Hence, root finding and fixed-point
resolution are two different views of the same problem.
See the :ref:`root finding <root_finding>` section for more details.

Accelerating JAXopt optimizers
------------------------------

Many optimizers can benefit from Anderson acceleration.  
Indeed, the root :math:`x` of a function :math:`F` is the fixed point of iterative root finding algorithms. 
Similarly the optimum :math:`x` of a function :math:`f` is the fixed point of iterative optimization algorithms.

To spare the user the burden of implementing Anderson acceleration for every solver, we propose the ``AndersonWrapper`` class.  
It takes an optimizer as input and applies Anderson acceleration to its iterates.

.. autosummary::
  :toctree: _autosummary

    jaxopt.AndersonWrapper

Its usage is transparent::

  gd = jaxopt.GradientDescent(fun=ridge_reg_objective, maxiter=500, tol=1e-3)
  aa = jaxopt.AndersonWrapper(solver=gd, history_size=5)
  sol, aa_state = aa.run(init_params, l2reg=l2reg, X=X, y=y)
  print(sol)

.. topic:: Examples

  * :ref:`sphx_glr_auto_examples_fixed_point_plot_anderson_wrapper_cd.py`
