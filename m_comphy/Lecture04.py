import numpy as np
from matplotlib import pyplot as plt

# calculation conditions
nx =40 
ny =40 
nx1 = nx+1
ny1 = ny+1
nt  = 2000
eps=1.e-5
xmin = 0
xmax = 4
ymin = 0
ymax = 4

omega = 1.5
 
dx = (xmax - xmin)/nx
dy = (ymax - ymin)/ny
 
# initial conditions
p  = np.zeros((nx1, ny1))
pnew = np.zeros((nx1, ny1))
b  = np.zeros((nx1, ny1))
ex = np.zeros((nx1, ny1))
ey = np.zeros((nx1, ny1))
b  = np.zeros((nx1, ny1))
x  = np.linspace(xmin, xmax, nx1)
y  = np.linspace(ymin, ymax, ny1)
 
# source term
b[int(nx/4),int(ny/4)] = -100
b[int(3*nx/4),int(3*ny/4)] = 100

# start iteration
for it in range(nt):
 
    p = pnew.copy()
    #change to SOR method
    for i in range(1, nx):
        for j in range(1, ny):
            pnew_gs = ((p[i+1,j] + pnew[i-1,j]) * dy * dy + (p[i,j+1] + pnew[i,j-1]) * dx * dx - b[i,j] * (dx * dx * dy * dy)) / (2 * (dx * dx + dy * dy))
            pnew[i,j] = (1 - omega) * p[i,j] + omega * pnew_gs

    #  boundary conditions
    for i in range(nx1):
        pnew[i,0] = pnew[i,1]
        pnew[i,ny1-1] = pnew[i,ny1-2]
    for j in range(ny1):
        pnew[0,j] = pnew[1,j]
        pnew[nx1-1,j] = pnew[nx1-2,j]

    dpmax = 0
    for i in range(1, nx):
        for j in range(1, ny):
            dp = abs(pnew[i,j] - p[i,j])
            dpmax = max(dpmax, dp)
    print(it, dpmax)
    if dpmax < eps:
        break
    
# post process calculate Ex Ey
for i in range(1,nx-1):
    for j in range(1,ny-1):
        ex[i,j] = (p[i+1,j]-p[i-1,j])/(2*dx)
        ey[i,j] = (p[i,j+1]-p[i,j-1])/(2*dy)

    
# draw contours, stream lines
cont=plt.contour(x,y,p,np.linspace(-0.0005,0.0005,21),colors='black')
cont.clabel(fmt='%1.1f', fontsize=14)
cont=plt.contour(x,y,p,np.linspace(-0.0005,0.0005,21),cmap='rainbow')
cont=plt.streamplot(x,y,ex,ey)
plt.show()
