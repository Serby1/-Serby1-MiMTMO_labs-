#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pprint
import math
import random
from scipy.linalg import expm
np.set_printoptions(precision=5)
pp=pprint.PrettyPrinter(indent=4)

np.set_printoptions(precision=5)
pp = pprint.PrettyPrinter(indent=4)


# # Вариант 10
# # Цепи с дискретным временем

# # 1

# In[2]:


P = np.array([[0.3, 0.7, 0, 0],
              [0.5, 0, 0.5, 0],
              [0, 0.5, 0, 0.5],
              [0, 0, 0.5, 0.5]])


# # 2

# In[3]:


for i in range(0, 50):
    pp.pprint(np.linalg.matrix_power(P, i))
    print("."*50)


# # Цепи с непрерывным временем

# # 3. Предельное распределение методом статических испытаний

# In[8]:


def exp_rand(lam):
    r=random.random()
    return -1/lam*math.log(r)

def next_state(current_state):
    r=random.random()
    if(current_state==0):
        if(r<=0.3):
            return 1
        else: 
            return 2
    elif(current_state==1):
        if(r<=0.5):
            return 0
        else:
            return 2
    elif(current_state==2):
        if(r<=0.5):
            return 1
        else:
            return 3
    else:
        if(r<=0.5):
            return 2
        else:
            return 1


# In[9]:



t_max=10**6
t_current=0
q=np.array([1,1,1,1])
sojourn_times=np.array([0, 0, 0, 0], dtype=np.float)
current_state=0
sojourn_times[0]=exp_rand(q[0])


while(t_current<t_max):
    current_state = next_state(current_state)
    time_in_state = exp_rand(q[current_state])
    sojourn_times[current_state]+=time_in_state
    t_current+=time_in_state
print(sojourn_times/sojourn_times.sum())
print((sojourn_times/sojourn_times.sum()).sum())


# # 2. Предельное распределение

# In[10]:


Q = np.array([[-1, 0.3, 0.7, 0],
              [0.5, -1, 0.5, 0],
              [0, 0.5, -1, 0.5],
              [0, 0.5, 0.5, -1]])
expm(Q*1000)


# In[ ]:





# In[ ]:




