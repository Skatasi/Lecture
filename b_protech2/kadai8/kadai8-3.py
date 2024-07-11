import numpy as np
from matplotlib import pyplot as plt
f_s = 256
N = 30
def main():
    X = np.zeros([3, f_s]) 
    x=[]
    t=[]
    for i in range(f_s):
        X[0][i] = 0.5*np.cos(2*np.pi*25*i/f_s + 0.5*np.pi)
        X[1][i] = 0.8*np.cos(2*np.pi* 5*i/f_s -     np.pi)
        X[2][i] = 1.0*np.cos(2*np.pi* 1*i/f_s + 0.2*np.pi)          
        x.append(X[0][i]+X[1][i]+X[2][i])
        t.append(i)
    a=[]
    b=[]
    for i in range(N):
        a.append(2/f_s*sum(x[j]*np.cos(2*np.pi*i*1*t[j]/f_s) for j in range(f_s)))
        b.append(2/f_s*sum(x[j]*np.sin(2*np.pi*i*1*t[j]/f_s) for j in range(f_s)))

    x_inv =[]
    for i in range(f_s):
        x_inv.append(a[0]/2 + sum(a[j]*np.cos(2*np.pi*i*t[j]/f_s) + b[j]*np.sin(2*np.pi*i*t[j]/f_s) for j in range(N)))


    
if __name__ == "__main__":
    main()