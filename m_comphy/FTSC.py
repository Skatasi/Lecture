import numpy as np
import matplotlib.pyplot as plt

# mesh number
Num_stencil_x = 18
x_array = np.arange(Num_stencil_x)

# set initial u_{0}
boundary_l = 1.0
boundary_r = 0.0
u_array = np.where((x_array > 5), boundary_r, boundary_l)

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

total_movement = C * Delta_t * (Time_step+1)
exact_u_array = np.where((x_array >= 5 + total_movement), boundary_r, boundary_l)

u_ftcs = u_array.copy()

# start time-marching method for FTCS
for n in range(Time_step):
    u_old = u_ftcs.copy()
    for j in range(1,Num_stencil_x-1):
        u_ftcs[j] = u_old[j] - 0.5 * CFL * (u_old[j+1] - u_old[j-1])

# post processing 
plt.plot(x_array, exact_u_array, label="Exact")
plt.plot(x_array, u_ftcs, label="FTCS")
plt.legend(loc="upper right")
plt.xlabel("x")
plt.ylabel("u(FTCS)")
plt.xlim(0, max(x_array))
plt.ylim(-0.2,1.8)

plt.show()