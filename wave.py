import numpy as np

np.set_printoptions(precision=3)

###################################
f = lambda x: np.sin(np.pi*x )
g = lambda x: 0
n = 5 # n 등분 
k = 5
###################################

h = 1 / n

u = np.empty(shape=(n+1, k+1))

u[:,0] = f(np.linspace(0,1,n+1))
u[0,:] = 0
u[n,:] = 0

u[1:-1,1] = 0.5*(u[:-2,0]+u[2:,0]) + h*g(np.linspace(h, 1-h, n-1))

for i in range(2,k+1):
    u[1:-1,i] = u[:-2,i-1] + u[2:,i-1] - u[1:-1,i-2]

print(u.T)
