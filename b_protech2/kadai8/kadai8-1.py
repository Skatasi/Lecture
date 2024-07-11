import numpy as np
from matplotlib import pyplot as plt
f_s = 256

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
    plt.figure(1); plt.clf()
    plt.subplot(211)
    plt.plot(t, X.T)
    plt.subplot(212)
    plt.plot(t, x)
    plt.xlabel('Time [s]')
    plt.show()

if __name__ == "__main__":
    main()