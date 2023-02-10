Line search
===========

Given current parameters :math:`x_k` and a descent direction :math:`p_k`,
the goal of a line search method is to find a step size :math:`\alpha_k`
such that the one-dimensional function

.. math::

  \varphi(\alpha_k) \triangleq f(x_k + \alpha_k p_k)

is minimized or at least a sufficient decrease is guaranteed.

Sufficient decrease and curvature conditions
--------------------------------------------

Exactly minimizing :math:`\varphi` is often computationally costly.
Instead, it is often preferred to search for :math:`\alpha_k` satisfying certain conditions.
One example of these conditions are the **Wolfe conditions**

.. math::

    \begin{aligned}
    f(x_k + \alpha_k p_k) &\le f(x_k) + c_1 \alpha_k \nabla f(x_k)^\top p_k \\
    \nabla f(x_k + \alpha_k p_k)^\top p_k &\ge c_2 \nabla f(x_k)^\top p_k
    \end{aligned}

where :math:`0 < c_1 < c_2 < 1`. These conditions are explained in greater detail in
Nocedal and Wright, see equations (3.6a) and (3.6b) there.

A step size may satisfy the Wolfe conditions without being particularly close
to a minimizer of :math:`\varphi` (Nocedal and Wright, Figure 3.5).  The
curvature condition in the second equation can be modified to force the step
size to lie in at least a broad neighborhood of a stationary point of
:math:`\varphi`. Combined with the sufficient decrease condition in the first
equation, these are known as the **strong Wolfe conditions**

.. math::

    \begin{aligned}
    f(x_k + \alpha_k p_k) &\le f(x_k) + c_1 \alpha_k \nabla f(x_k)^\top p_k \\
    |\nabla f(x_k + \alpha_k p_k)^\top p_k| &\le c_2 |\nabla f(x_k)^\top p_k|
    \end{aligned}

where again :math:`0 < c_1 < c_2 < 1`. See Nocedal and Wright, equations (3.7a) and (3.7b).

Algorithms
----------

.. autosummary::
  :toctree: _autosummary

    jaxopt.BacktrackingLineSearch
    jaxopt.HagerZhangLineSearch

The :class:`BacktrackingLineSearch <jaxopt.BacktrackingLineSearch>` algorithm
iteratively reduces the step size by some decrease factor until the conditions
above are satisfied. Example::

    ls = BacktrackingLineSearch(fun=fun, maxiter=20, condition="strong-wolfe",
                                decrease_factor=0.8)
    stepsize, state = ls.run(init_stepsize=1.0, params=params,
                             descent_direction=descent_direction,
                             value=value, grad=grad)

where

* ``init_stepsize`` is the first step size value to try,
* ``params`` are the current parameters :math:`x_k`,
* ``descent_direction`` is the provided descent direction :math:`p_k` (optional, defaults to :math:`-\nabla f(x_k)`),
* ``value`` is the current value :math:`f(x_k)` (optional, recomputed if not provided),
* ``grad`` is the current gradient :math:`\nabla f(x_k)` (optional, recomputed if not provided),

The returned ``state`` contains useful information such as ``state.params``,
which contains :math:`x_k + \alpha_k p_k` and ``state.grad``, which contains
:math:`\nabla f(x_k + \alpha_k p_k)`.

.. topic:: References:

 * Numerical Optimization, Jorge Nocedal and Stephen Wright, Second edition.
