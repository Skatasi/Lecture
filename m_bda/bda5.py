import pandas as pd

db = {'department':[0,0,0,1,1,1,1,2,2,3,3],
      'status':[1,0,0,0,1,0,1,1,0,1,0],
      'age':[2,1,2,0,2,1,4,3,2,5,1],
      'salary':[4,0,1,4,5,4,5,4,3,2,0],
      'count':[30,40,40,20,5,3,3,10,4,4,6]}
N=sum(db['count'])

p_C = [0]*2
N_C = [0]*2
p_d1C = [0]*2
p_a2C = [0]*2
p_s4C = [0]*2

for i in range(11):
    N_C[db['status'][i]] += db['count'][i]

for i in range(11):
    if db['department'][i] == 1:
        p_d1C[db['status'][i]] += db['count'][i]/N_C[db['status'][i]]
    if db['age'][i] == 2:
        p_a2C[db['status'][i]] += db['count'][i]/N_C[db['status'][i]]
    if db['salary'][i] == 4:
        p_s4C[db['status'][i]] += db['count'][i]/N_C[db['status'][i]]

print(N_C[0]/sum(N_C), N_C[1]/sum(N_C))
print(p_d1C, p_a2C, p_s4C)
print(p_d1C[0]*p_a2C[0]* p_s4C[0])
print(p_d1C[0]*p_a2C[0]* p_s4C[0]*N_C[0]/sum(N_C))
print(p_d1C[1]*p_a2C[1]* p_s4C[1])
print(p_d1C[1]*p_a2C[1]* p_s4C[1]*N_C[1]/sum(N_C))