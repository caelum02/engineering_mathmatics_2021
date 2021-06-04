import numpy as np
from solver import gauss_seidel

np.set_printoptions(precision=3)

#################
n = 5
r = 1.
f = lambda x: np.sin(np.pi*x)
steps = 5
#################

h = 1 / n
k = r * h**2

u = np.zeros(shape=(n+1, steps+1))
u[:,0] = f(np.linspace(0, 1, n+1))

A = np.zeros(shape=(n-1, n-1))
A += (2+2*r) * np.eye(n-1)
A[1:,:-1] += -r * np.eye(n-2)
A[:-1,1:] += -r * np.eye(n-2)

for i in range(1,steps+1):
    b = (2-2*r) * u[1:-1,i-1] + r * (u[2:,i-1] + u[:-2,i-1])
    u[1:-1,i] = gauss_seidel(A, b)

print(u)