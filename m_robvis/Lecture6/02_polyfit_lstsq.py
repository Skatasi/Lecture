import numpy as np
import matplotlib.pyplot as plt


def linear_lst_squares_normaleq(A, b):
    '''
    Solves Ax=b for x by least-squares using normal equation
    '''
    return np.linalg.inv(A.T@A)@A.T@b


def linear_lst_squares_numpy(A, b):
    '''
    Solves Ax=b for x by least-squares using numpy built-in function
    '''
    return np.linalg.lstsq(A, b, rcond=None)[0]


# data points
a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([0.0, 1.4, 2.0, 3.1, 4.0, 5.0])

# Polynomial model f(x) = b = x1 + x2 * a + ... + xp * a^(p-1)
p = 4
# set up a linear system A x = b
x = np.empty((p, 1))
A = np.ones(a.shape[0])
for i in range(1, p):
    A = np.vstack((A, a**i))
A = A.T
b = np.array([b]).T

x = linear_lst_squares_normaleq(A, b)
#x = linear_lst_squares_numpy(A, b)

a_range = np.linspace(-1., 6., 100)
plt.plot(a, b, 'o')
plt.plot(a_range, np.polyval(np.flipud(x), a_range), 'g-')
plt.xlabel('a')
plt.ylabel('b')
plt.show()
