import numpy as np
import matplotlib.pyplot as plt


def linear_lst_squares_normaleq(A, b):
    '''
    Solves Ax=b for x by least-squares using normal equation
    '''
#    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)   # Python 2.x notation
    return np.linalg.inv(A.T@A)@A.T@b   # Python 3 replaces .dot with @

def linear_lst_squares_numpy(A, b):
    '''
    Solves Ax=b for x by least-squares using numpy built-in function
    '''
    return np.linalg.lstsq(A, b, rcond=None)[0]


# data points
a = np.array([0.0, 1.0, 2.0, 3.0,  4.0, 5.0])
b = np.array([0.0, 1.4, 2.0, 3.1,  4.0, 5.0])

# Line model f(x1, x2) = b = x1 + x2 * a
# set up a linear system A x = b
x = np.empty((2, 1))
A = np.vstack((np.ones(a.shape[0]), a)).T
b = np.array([b]).T


x = linear_lst_squares_normaleq(A, b)
#x = linear_lst_squares_numpy(A, b)

a_range = [-1., 6.]
plt.plot(a, b, 'o')
plt.plot(a_range, x[0] + x[1] * a_range, 'g-')
plt.xlabel('a')
plt.ylabel('b')
plt.show()
