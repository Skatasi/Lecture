import numpy as np
import matplotlib.pyplot as plt


def L1_residual_min(A, b, MAX_ITER=1000, tol=1.0e-8):
    """
    L1 residual minimization by iteratively reweighted least squares (IRLS)
        minimize ||Ax - b||_1
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")
    eps = 1.0e-8
    m, n = A.shape

    xold = np.ones((n, 1))
    W = np.identity(m)
    if np.ndim(b) != 2 and b.shape[1] != 1:
        raise ValueError("b needs to be a column vector m x 1")

    iter = 0
    while iter < MAX_ITER:
        iter = iter + 1
        # Solve the weighted least squares WAx=Wb
        x = np.linalg.lstsq(W @ A, W @ b, rcond=None)[0]
        r = b - A @ x
        # Termination criterion
        if np.linalg.norm(x - xold) < tol:
            return x
        else:
            xold = x
        # Update weighting factor
        W = np.diag(np.asarray(1.0 / np.maximum(np.sqrt(np.fabs(r)), eps))[:, 0])
    return x


def linear_lst_squares_numpy(A, b):
    '''
    Solves Ax=b for x by least-squares using numpy built-in function
    '''
    return np.linalg.lstsq(A, b, rcond=None)[0]


# data points
a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([0.0, 3.0, 2.0, 3.1, 4.0, 5.0])

# Polynomial model f(x) = b = x1 + x2 * a + ... + xp * a^(p-1)
p = 2  # a line
# set up a linear system A x = b
theta = np.empty((p, 1))
A = np.ones(a.shape[0])
for i in range(1, p):
    A = np.vstack((A, a**i))
A = A.T
b = np.array([b]).T

x_L1 = L1_residual_min(A, b)
x_L2 = linear_lst_squares_numpy(A, b)

a_range = np.linspace(-1., 6., 100)
plt.plot(a, b, 'o')
plt.plot(a_range, np.polyval(np.flipud(x_L1), a_range), 'g-', label="L1")
plt.plot(a_range, np.polyval(np.flipud(x_L2), a_range), 'r-', label="L2")
plt.legend(loc="upper left")
plt.xlabel('a')
plt.ylabel('b')
plt.show()
