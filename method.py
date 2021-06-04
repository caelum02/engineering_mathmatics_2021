import numpy as np
from numba import njit

dtype = np.float64

class Method():
    def __init__(self, f, y_0, h, solver, x_0=0):
        self.f = f
        self.x_0 = x_0
        self.h = h
        self.solver = solver

        if type(y_0) is type(float) or type(y_0) is int or isinstance(type(y_0), np.dtype):
            self.y_0 = np.array(y_0, dtype=dtype)
        elif type(y_0) is np.ndarray:
            self.y_0 = y_0
        else:
            raise TypeError

    def solve(self, num_step=None, x=None, isRKF=False):
        
        if num_step is None:
            num_step = (x - self.x_0) // self.h

        if x is None:
            x = self.x_0 + num_step * self.h
        
        
        x = np.linspace(self.x_0, x, num=num_step+1)
        y = np.zeros(shape=(len(x), len(self.y_0)))
        y[0,:] = self.y_0

        
        if isRKF:
            return self._rkf_core(num_step, x, y, self.h, self.solver)
        else:
            return self._core(num_step, x, y, self.h, self.solver)

    @staticmethod
    @njit
    def _core(num_step, x, y, h, solver):
        for i in range(num_step):
            y[i+1,:] = y[i,:] + solver(x[i], y[i,:], h)

        return x, y, None

    @staticmethod
    @njit
    def _rkf_core(num_step, x, y, h, solver):
        err = np.zeros_like(y)
        for i in range(num_step):
            dy, e = solver(x[i], y[i,:], h)
            y[i+1,:] = y[i,:] + dy
            err[i+1,:] = e

        return x, y, err