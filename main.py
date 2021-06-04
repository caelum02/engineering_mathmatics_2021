from method import *
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import time

dtype = np.float64

@njit
def f(x, y):
    return 1-x+4*y

@njit
def Euler(x, y, h):
    return f(x, y)

@njit
def ImprovedEuler(x, y, h):
    k = f(x, y)
    return h * 0.5 * (k + f(x, y + h * k))

@njit
def ClassicRK4(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)
    return (k1 + 2*k2 + 2*k3 + k4) / 6

@njit
def RKF(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/4, y + k1/4)
    k3 = h * f(x + 3/8*h, y + 3/32*k1 + 9/32*k2)
    k4 = h * f(x + 12/13*h, y + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)
    k5 = h * f(x + h, y + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4)
    k6 = h * f(x + 1/2*h, y - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)
    
    y_ = 16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6
    e = k1/360 - 128/4275*k3 - 2197/75240*k4 + 1/50*k5 + 2/55*k6

    return y_, e

x=0.4
y=5.8388
for i in range(5):
    y += ClassicRK4(x,y,0.2)
    print(y)


# method = Method(f, np.array([1]), 0.1, ClassicRK4, x_0=1)


# x, y, err = method.solve(num_step=10, isRKF=False)
# for i, j in zip(x,y):
#     print(f'y({i:.1f}) = {j[0]:.6f}')

# plt.plot(x, y)
# plt.plot(x, np.exp(x**2-1))
# plt.show()

# print(y, np.exp(x**2-1))
# # plt.plot(x, y-err)?
# # print(np.abs(err)[:-1])

