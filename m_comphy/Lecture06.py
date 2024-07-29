import numpy as np
import matplotlib.pyplot as plt

# mesh number
Num_stencil_x = 18
x_array = np.arange(Num_stencil_x+1)

# set initial u_{0}
boundary_l = 1.0
boundary_r = 0.0
u_array = np.where((x_array > 5), boundary_r, boundary_l)
g_array = np.zeros(Num_stencil_x+1)

# time step 
Time_step = 6

# Delta x 
Delta_x = max(x_array) / (Num_stencil_x-1)

# velocity
C = 1
# time step
Delta_t = 0.5
# CFL number
CFL = C * Delta_t / Delta_x
xi = -C * Delta_t

total_movement = C * Delta_t * (Time_step+1)
exact_u_array = np.where((x_array >= 5 + total_movement), boundary_r, boundary_l)

u_cip = u_array.copy()

# start time-marching method for cip
for n in range(Time_step):
    u_old = u_cip.copy()
    g_old = g_array.copy()

    a = ((g_old[1:-1] + g_old[:-2]) / (Delta_x**2) 
         - 2 * (u_old[1:-1] - u_old[:-2]) / (Delta_x**3))
    b = (3 * (u_old[:-2] - u_old[1:-1]) / (Delta_x**2) 
         + (2 * g_old[1:-1] + g_old[:-2]) / Delta_x)
    u_cip[1:-1] = (a * (xi**3) + b * (xi**2) 
               + g_old[1:-1] * xi + u_old[1:-1])
    g_array[1:-1] = (3 * a * (xi**2) + 2 * b * xi 
               + g_old[1:-1])

# post processing 
plt.plot(x_array, exact_u_array, label="Exact")
plt.plot(x_array, u_cip, label="cip")
plt.legend(loc="upper right")
plt.xlabel("x")
plt.ylabel("u(cip)")
plt.xlim(0, max(x_array))
plt.ylim(-0.2,1.8)

plt.show()
