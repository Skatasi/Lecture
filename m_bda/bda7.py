import numpy as np

data = np.array([[0,0],[2,0],[2,1],[3,0],[4,1],[5,0],[6,0],[8,0]])
flag = [0] * len(data)
eps = 2
min = 3
cluster = []
for i in range(len(data)):
    if flag[i] == 0:
        flag[i] = 1
        c = []
        continue
    for j in range(len(data)):
        if flag[j] == 0:
            continue
        if np.linalg.norm(data[i]-data[j],1) < eps:
            c.append(data[j])
            flag[j] = 1
    if len(cluster) >= min:
        print('cluster:',cluster)
    else:
        print('noise:',data[i])

print(cluster)