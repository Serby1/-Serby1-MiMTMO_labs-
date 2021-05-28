#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import pprint
import math
import random
from scipy.linalg import expm
import math as m
np.set_printoptions(precision=5)
pp=pprint.PrettyPrinter(indent=4)


# # Вариант 10

# # Задача 1

# In[2]:


P = np.array([[0, 0.3, 0.7, 0, 0, 0],
              [0, 0, 0.6, 0.4, 0, 0],
              [0, 0, 0, 0.2, 0.8, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0.1, 0, 0.9],
              [1, 0, 0, 0, 0, 0]])

mu = [12, 25, 32, 45, 56]

def stationary_distribution(matrix, eps):
    L=matrix.shape[0]-1
    omega_0 = np.ones(matrix.shape[0])/(L+1)
    omega_current = omega_0 @ matrix
    while(np.linalg.norm(omega_0 - omega_current)>eps):
        omega_0=omega_current
        omega_current = omega_current @ matrix
    return omega_current

def get_lamds(lamda_0, omega):
    lamda = lamda_0/omega[0]*omega
    lamda = lamda[1:P.shape[0]] 
    return lamda

def get_n(lamda, mu):
    return lamda/(mu-lamda)
    
def get_u(lamda, mu):
    return 1/(mu-lamda)

def get_psi(lmbds, kappa, mu):
    psi = np.zeros(L)
    for i in range(L):
        psi[i] = lmbds[i]/(kappa[i]*mu[i])
    return psi


def get_tau(L, l0, k, psi, lambdas):
    Pi0 = np.zeros(L)
    bi = np.zeros(L)
    hi = np.zeros(L)
    ni = np.zeros(L)
    ui = np.zeros(L)
    tau = 0
    for i in range(L):
        _sum = 0
        for j in range(int(k[i])):
            _sum += (k[i]*psi[i])**j / m.factorial(j)
        Pi0[i] = ((((k[i]*psi[i])**k[i])/
                       (m.factorial(k[i])*(1-psi[i])))+_sum)**(-1)
        bi[i] = Pi0[i]*(((k[i]**k[i])*(psi[i]**(k[i]+1)))/
                            (m.factorial(k[i])*((1-psi[i])**2)))
        hi[i] = psi[i]*k[i]
        ni[i] = bi[i]+hi[i]
        ui[i] = ni[i]/lambdas[i]
        tau += lambdas[i]*ui[i]
    tau = (1/l0)*tau
    
    return tau




# In[3]:


omega=stationary_distribution(P, 0.001)
print("omega=",omega)


# In[4]:


lamda_array=[]
n_s_array=[]
u_s_array=[]
for lamda_0 in range(1, 14, 3):
    lamda = get_lamds(lamda_0, omega)
    n_s = get_n(lamda, mu)
    u_s = get_u(lamda, mu)
    lamda_array.append(lamda_0)
    n_s_array.append(n_s)
    u_s_array.append(u_s)


# In[5]:


u_s_array=np.array(u_s_array)
u_s_array=u_s_array.transpose()
u_s_array


# In[6]:


n_s_array=np.array(n_s_array)
n_s_array=n_s_array.transpose()
n_s_array


# In[7]:


for s, n_array in enumerate(n_s_array):
    plt.plot(lamda_array, n_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(f'Зависимость мат. ожидания числа требований\nв системах от интенсивности поступающих в сеть требований')
print()
plt.show()


# In[8]:


for s, u_array in enumerate(u_s_array):
    plt.plot(lamda_array, u_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(
    f'Зависимость мат. ожидания длительностей пребывания требований\nв системах от интенсивности поступающих в сеть требований')
print()
plt.show()


# In[9]:


mu_array=[]
n_s_array=[]
u_s_array=[]
for i in range(100, 200, 20):
    mu[1] = i
    n_s = get_n(lamda, mu)
    u_s = get_u(lamda, mu)
    n_s_array.append(n_s)
    u_s_array.append(u_s)
    mu_array.append(mu[1])
    
   


# In[10]:


u_s_array=np.array(u_s_array)
u_s_array=u_s_array.transpose()
u_s_array


# In[11]:


n_s_array=np.array(n_s_array)
n_s_array=n_s_array.transpose()
n_s_array


# In[12]:


for s, n_array in enumerate(n_s_array):
    plt.plot(mu_array, n_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(f'Зависимость мат. ожидания числа требований\nв системах от интенсивности обслуживания в 2-й системе')
print()
plt.show()


# In[13]:


for s, u_array in enumerate(u_s_array):
    plt.plot(mu_array, u_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(
    f'Зависимость мат. ожидания числа требований\nв системах от интенсивности обслуживания в 2-й системе')
print()
plt.show()


# # Задача 2

# ## A) Нахождение маршрутной матрицы с максимальной пропускной способностью¶
# 
# 

# In[14]:


L = 5
lambda_0=0.5
kappa = np.ones(L) 
mu = np.ones(L) 
P = np.array([
    [0,       0.4,    0,      0.6,    0,      0],
    [0,       0,      0.5,    0,      0.5,    0],
    [0,       0,      0.9,    0,      0,      0.1],
    [0,       0,      0,      0,      1,      0],
    [0.2,     0,      0,      0,      0,      0.8],
    [1,       0,      0,      0,      0,      0]
]) #начальная маршрутная матрица

print(f'Начальная маршрутная матрица:\n{P}')


# In[15]:


d = 0.1
leng_arr = np.arange(0.1,1,d)
mmax_start = 0
mmax_end = 0



for i0 in leng_arr:
    for i1 in leng_arr:
        for i2 in leng_arr:
            for i4 in leng_arr:
                                
                P[0,1] = i0
                P[0,3] = 1-i0
                                
                P[1,2] = i1
                P[1,4] = 1-i1
                                
                P[2,2] = i2
                P[2,5] = 1-i2

                P[4,0] = i4
                P[4,5] = 1-i4
                                
                omega = stationary_distribution(P, 0.001)
                templ = get_lamds(lambda_0, omega)
                tempp = get_psi(templ, kappa, mu)
                if (all(tempp < 1)) and (sum(templ) > mmax_start):
                    mmax_start = sum(templ)
                    m_lambdas = np.copy(templ)
                    th_start = np.copy(P)
                elif (any(tempp > 1)) and (sum(templ) > mmax_end):
                    mmax_end = sum(templ)
                    m_lambdas_start = np.copy(templ)
                    th_end = np.copy(P)

print('Изначальная пропускная способность сети:', mmax_start)
print('Маршрутная матрица:\n', th_start)
print('\n\n')
print('Максимальная пропускная способность:', mmax_end)
print('Маршрутная матрица:\n', th_end)


# ## Б)Нахождение вектора интенсивностей обслуживания при заданном значении пропускной способности сети¶

# In[16]:


L = 5
Q = 3
kappa = np.ones(L) 
lambda_0 = 0.5
mu= np.ones(L) # интенсивности обслуживания в i-ой системе
mu = mu/10
P = np.array([[0, 0.3, 0.7, 0, 0, 0],
              [0, 0, 0.6, 0.4, 0, 0],
              [0, 0, 0, 0.2, 0.8, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0.1, 0, 0.9],
              [1, 0, 0, 0, 0, 0]]) #начальная маршрутная матрица

omega = stationary_distribution(P, 0.001)
lambdas = get_lamds(lambda_0, omega)
psi = get_psi(lambdas, kappa, mu)
tau = get_tau(L, lambda_0, kappa, psi, lambdas)
index = 0

# приближаемся к значению tau = Q
while m.fabs(tau - Q) > 0.01:
    # меняем элементы вектора по очереди на 0.01
    mu[index % 5] += 0.01
    index += 1
    psi = get_psi(lambdas, kappa, mu)
    tau = get_tau(L, lambda_0, kappa, psi, lambdas)

print("Вектор mu: ", mu)
print(f"Сумма вектора mu = {sum(mu)}")
print('Q = ', Q)
print("Время реакции: ", tau)
print("Маршрутная матрица:\n", P)


# In[ ]:




