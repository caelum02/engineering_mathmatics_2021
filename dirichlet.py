import numpy as np
import solver
from numba import njit

def boundary_left(x, y):
    return 100

def boundary_right(x, y):
    return 100

def boundary_top(x, y):
    return 0

def boundary_bottom(x, y):
    return 100

def f(x, y):
    return 0

def initialize(G, boundary_left, boundary_right, boundary_bottom, boundary_top, x_left, y_bottom, x_right, y_top, m, n, h):
    x = np.linspace(x_left, x_right, m+1)[1:-1]
    y = np.linspace(y_bottom, y_top, n+1)[1:-1]

    G[1:-1,0] = boundary_bottom(x, y_bottom)
    G[1:-1,-1] = boundary_top(x, y_top)
    G[0,1:-1] = boundary_left(x_left, y)
    G[-1,1:-1] = boundary_right(x_right, y)

# mxn matrix
h = 4.
x_left = 0.
y_bottom = 0.
x_right = 12.
y_top = 12.

m = int((x_right - x_left) / h)
n = int((y_top - y_bottom) / h)
print(m, n)

# G = np.zeros(shape=(m+1, n+1))
# initialize(G, boundary_left, boundary_right, boundary_bottom, boundary_top, x_left, y_bottom, x_right, y_top, m, n, h)

num_var = (m-1)*(n-1)
A = np.empty(shape=(num_var, num_var))
b = np.empty(shape=(num_var,))

for k in range(num_var):
    i = k // (n-1)
    j = k % (n-1)
    x, y = (i+1)*h, (j+1)*h

    print(i, j, k)

    A[k,k] = -4.
    b[k] = f(x, y) * h**2
    
    if i==0:
        b[k] -= boundary_left(x_left, y)
    else:
        A[k, k-(n-1)] = 1
    
    if i==m-2:
        b[k] -= boundary_right(x_right, y)
    else:
        A[k, k+(n-1)] = 1

    if j==0:
        b[k] -= boundary_bottom(x, y_bottom)
    else:
        A[k, k-1] = 1

    if j==n-2:
        b[k] -= boundary_top(x, y_top)
    else:
        A[k, k+1] = 1
    

print(A, b)

print(solver.gauss_seidel(A, b, 1000))
print(solver.jacobi(A, b, 1000))