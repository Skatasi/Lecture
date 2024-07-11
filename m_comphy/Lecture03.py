import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 
th = np.arange(0,1*np.pi,0.01)
#
for mu in (-1/6,1/6,1/3,1/2,2/3,1):
    gg = mu*2*(np.cos(th)-1)+1                                  # input function here  g(mu, np.cos(th))
    plt.plot(th, gg, label="mu= %1.3f" %mu)
# 
plt.xlim(0, 1*np.pi)
plt.ylim(-2,2)
plt.hlines(-1 ,0 , 1*np.pi, "k", linestyle=":")
plt.hlines( 1 ,0 , 1*np.pi, "k", linestyle=":")
plt.hlines( 0 ,0 , 1*np.pi, "k", linestyle="-")
plt.title('von Neumann stability analysis',fontsize=15)
plt.xlabel(r'$\theta$',fontsize=14)
plt.ylabel('g',fontsize=14)
plt.legend()
plt.savefig("graph.png")
plt.show()