import numpy as np
import qpsolvers as qp

M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M) # this is a positive definite matrix
q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([1., 1., 1.]).reshape((3,))
A = np.array([1., 1., 1.])
b = np.array([1.])

x = qp.solve_qp(P, q, G, h, A, b, solver="quadprog")
print("QP solution: x = {}".format(x))
