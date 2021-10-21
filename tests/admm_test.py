"""Tests for admm."""
# pylint: disable=invalid-name

import cvxopt
from jaxopt._src import admm
import numpy as np
from jax import test_util as jtu
from absl.testing import absltest


def cvxopt_setup_qp(problem, ineq=None, eq=None):
  cvxopt.solvers.options['feastol'] = 1e-8
  cvxopt.solvers.options['abstol'] = 1e-8
  cvxopt.solvers.options['reltol'] = 1e-8
  cvxopt.solvers.options['show_progress'] = True

  ineq_mat = cvxopt.matrix(ineq[0]) if ineq else None
  ineq_vec = cvxopt.matrix(ineq[1]) if ineq else None
  eq_mat = cvxopt.matrix(eq[0]) if eq else None
  eq_vec = cvxopt.matrix(eq[1]) if eq else None

  soln = cvxopt.solvers.qp(
      cvxopt.matrix(problem[0]),
      cvxopt.matrix(problem[1]),
      G=ineq_mat,
      h=ineq_vec,
      A=eq_mat,
      b=eq_vec)
  return np.squeeze(soln['x'])


class AdmmTest(jtu.JaxTestCase):

  def test_qp(self):
    # Setup a random QP min_x 0.5*x'*Q*x + q'*x s.t. A*x = b, x>=0
    problem_size = 16
    constraints = 2
    P = np.random.randn(problem_size, problem_size)
    P = P.T.dot(P)
    q = np.random.randn(problem_size)
    A = np.random.randn(constraints, problem_size)
    b = np.random.randn(constraints)

    cvxopt_soln = np.array(
        cvxopt_setup_qp((P, q),
                        (-np.eye(problem_size), np.zeros(
                            (problem_size, 1))), (A, b)))
    x0 = np.zeros(P.shape[0])
    rho = 1.0
    maxiter = 10000
    admm_soln = admm.qp(admm.factorize(P, A, rho), q, b, x0, rho, maxiter)[0]

    self.assertLess(np.linalg.norm(admm_soln - cvxopt_soln), 0.1)

if __name__ == '__main__':
  absltest.main()
