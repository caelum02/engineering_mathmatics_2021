import numpy as np
from numba import njit

def gauss_seidel(A, b, k=10000):
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    L_ = np.tril(A)
    U = np.triu(A, k=1)
    L_inv = np.linalg.inv(L_)
    x = np.zeros_like(b)

    for _ in range(k):
        x = np.dot(L_inv, b-np.dot(U, x))

    return x

def jacobi(A, b, k):
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    
    diag = np.diag(A).copy()
    np.fill_diagonal(A, 0)
    x = np.zeros_like(b)
    
    for i in range(k):
        x = (b - np.dot(A, x)) / diag
        print(x)

    return x

def newton(f, fdot, x0, n):
    x = x0
    for i in range(n):
        x = x - f(x) / fdot(x)
    
    return x

def secant(f, x0, x1, n):
    x = x1
    x_ = x0
    for i in range(n):
        tmp  = x - f(x) * (x-x_) / (f(x)-f(x_))
        x_ = x
        x = tmp

    return x

if __name__ == '__main__':
    # np.set_printoptions(precision=3)
    # n = int(input("number of variables : "))
    # A = np.empty(shape=(n, n))
    # b = np.empty(shape=(n,))
    
    # print("A :")
    # for i in range(n):
    #     A[i,:] = list(map(float, input().split()))
    # print("b :")
    # b[:] = list(map(float, input().split()))
    
    # # print(gauss_seidel(A, b, k=1000))
    # print(jacobi(A, b, 5))
    # # print(jacobi(A, b, 1000))
     
    f = lambda x: x - 2*np.sin(x)
    f_ = lambda x: 1 - 2*np.cos(x)

    print(secant(f, 2, 1.9, 3))
    print(newton(f, f_, 2, 3))