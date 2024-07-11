import numpy as np
import pandas as pd

data = np.array([[2,10],[2,5],[8,4],[5,8],[7,5],[6,4],[1,2],[4,9]])
k = 3
initial = [0,3,6]
mean=np.array([data[0],data[3],data[6]])
flag = 1
while flag == 1:
    cluster = [[],[],[]]
    for i in range(8):
        dist = [np.linalg.norm(data[i]-mean[j]) for j in range(3)]
        cluster[dist.index(min(dist))].append(data[i])
        print('dist:',dist)
    new_mean = np.array([np.mean(cluster[i],axis=0) for i in range(3)])
    if np.all(new_mean == mean):
        flag = 0
    else:
        mean = new_mean
    print('mean',new_mean)
    print('cluster',cluster)

print(cluster)