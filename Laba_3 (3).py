#!/usr/bin/env python
# coding: utf-8

# # Задача 1

# In[1]:


# Вариант № 10 
import numpy as np
import matplotlib.pyplot as plt

# функция для расчета стационарного решения
# решение системы omega*theta=omega с условием нормировки sum(omega)=1
def stationary_distribution(matrix, eps):
    L=matrix.shape[0]
    omega_0 = np.ones(matrix.shape[0])/(L)
    omega_current = omega_0 @ matrix
    while(np.linalg.norm(omega_0 - omega_current)>eps):
        omega_0=omega_current
        omega_current = omega_current @ matrix
    return omega_current


# функция для расчета данных и вывода результатов для СеМО
def get_data(L, N, mu, theta, number):
    # решение уравнений omega*theta = omega с условием нормировки
    omega = stationary_distribution(theta,  0.001)
    # м.о. числа требований в системах
    s = [[0] * L for _ in range(N + 1)]
    # м.о. длительности пребывания требований в системах
    u = [[0] * L for _ in range(N + 1)]
    
    #рекурсивные метод анализа СеМО, начально условие s_i|0=0
    for Y in range(1, N + 1):
            for i in range(L):
                u[Y][i] = 1 / mu[i] * (s[Y-1][i] + 1)
            for i in range(L):
                summa = 0
                for j in range(L):
                    summa += omega[j] * u[Y][j]
                s[Y][i] = omega[i] * u[Y][i] * Y / summa
    
    # м.о. длительности ожидания требований в очереди системы
    w = [0] * L
    # м.о. числа требований, ожидающих обслуживание в очереди системы
    b = [0] * L
    # м.о. числа занятых приборов в системах
    h = [0] * L
    # интенсивность входящего потока требований в системы
    lambdas = [0] * L
    

    for i in range(L):
        w[i] = u[N][i] - (1 / mu[i])
        b[i] = s[N][i] * w[i] / u[N][i]
        h[i] = s[N][i] - b[i]
        lambdas[i] = h[i] * mu[i]
            
    return s[N], u[N], lambdas, mu[number - 1] 

# условия задачи
L = 5 # число приборов
N = 25# число требований в сети
mu = [0.8, 1, 1.2, 1.4, 0.6] # интенсивности обслуживания в i-ой системе
# маршрутная матрица
theta = np.array([
    [0,    0.6,  0.4,  0,    0],
    [0,    0,    0.4, 0.1,  0.5],
    [0,  0,    0.5,  0.5,    0],
    [0.5,    0,  0,    0,    0.5],
    [0.3,    0.7,    0,    0,  0]
]) 


# ## для 3-й системы

# In[2]:



s_s_array=[]
u_s_array=[]  
lambd_array=[]
number_mu = 3
mu_3 = np.arange(0.1, 3, 0.1)
for m in mu_3:
    mu[number_mu - 1] = m
    data = get_data(L, N, mu, theta, number_mu)
    s_s_array.append(data[0])
    u_s_array.append(data[1])
    lambd_array.append(data[2])
    

    


# In[3]:


u_s_array=np.array(u_s_array)
u_s_array=u_s_array.transpose()

s_s_array=np.array(s_s_array)
s_s_array=s_s_array.transpose()

lambd_array=np.array(lambd_array)
lambd_array=lambd_array.transpose()


# In[4]:


for s, s_array in enumerate(s_s_array):
    plt.plot(mu_3, s_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(f'Зависимость мат. ожидания числа требований\nв системах от интенсивности обслуживания в 3-й системе')
print()
plt.show()


# In[5]:


for s, u_array in enumerate(u_s_array):
    plt.plot(mu_3, u_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(
    f'Зависимость мат. ожидания длительности пребывания требований\nв системах от интенсивности обслуживания в 3-й системе')
print()
plt.show()


# In[6]:


for s, lambd in enumerate(lambd_array):
    plt.plot(mu_3, lambd, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(
    f'Зависимость интенсивности потоков требований, поступающих\nв системы от интенсивности обслуживания в 3-й системе')
print()
plt.show()


# ## для 5-ой системы

# In[7]:


s_s_array=[]
u_s_array=[]  
lambd_array=[]
number_mu = 5
mu_5 = np.arange(0.1, 3, 0.1)
for m in mu_5:
    mu[number_mu - 1] = m
    data = get_data(L, N, mu, theta, number_mu)
    s_s_array.append(data[0])
    u_s_array.append(data[1])
    lambd_array.append(data[2])
    


# In[8]:


u_s_array=np.array(u_s_array)
u_s_array=u_s_array.transpose()

s_s_array=np.array(s_s_array)
s_s_array=s_s_array.transpose()

lambd_array=np.array(lambd_array)
lambd_array=lambd_array.transpose()


# In[9]:


for s, s_array in enumerate(s_s_array):
    plt.plot(mu_5, s_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(f'Зависимость мат. ожидания числа требований\nв системах от интенсивности обслуживания в 5-й системе')
print()
plt.show()


# In[10]:


for s, u_array in enumerate(u_s_array):
    plt.plot(mu_5, u_array, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(
    f'Зависимость мат. ожидания длительности пребывания требований\nв системах от интенсивности обслуживания в 5-й системе')
print()
plt.show()


# In[11]:


for s, lambd in enumerate(lambd_array):
    plt.plot(mu_5, lambd, 'o-', label='S'+str(s+1))
plt.legend()
plt.title(
    f'Зависимость интенсивности потоков требований, поступающих\nв системы от интенсивности обслуживания в 5-й системе')
print()
plt.show()


# # Задача 2

# ## А)Необходимо подобрать такой вектор интенсивностей обслуживания, чтобы м.о. числа требований в системах были одинаковы

# In[3]:


# Вариант № 10 
import numpy as np
import matplotlib.pyplot as plt
import random

# условия задачи
L = 5 # число приборов
N = 25# число требований в сети
mu = [0.8, 1, 1.2, 1.4, 0.6] # интенсивности обслуживания в i-ой системе
# маршрутная матрица
theta = np.array([
    [0,    0.6,  0.4,  0,    0],
    [0,    0,    0.4, 0.1,  0.5],
    [0,  0,    0.5,  0.5,    0],
    [0.5,    0,  0,    0,    0.5],
    [0.3,    0.7,    0,    0,  0]
]) 

# функция для расчета стационарного решения
# решение системы omega*theta=omega с условием нормировки sum(omega)=1
def stationary_distribution(matrix, eps):
    L=matrix.shape[0]
    omega_0 = np.ones(matrix.shape[0])/(L)
    omega_current = omega_0 @ matrix
    while(np.linalg.norm(omega_0 - omega_current)>eps):
        omega_0=omega_current
        omega_current = omega_current @ matrix
    return omega_current


#функция для проверки равенства элементов
def checking_for_identical_elements(_list):
    delta = 0.01
    for i in range(1, len(_list)):
        if abs(_list[0] - _list[i]) > delta:
            return False
    return True

def get_s(L, N, mu, theta):
    # Решение уравнений omega*Theta = omega с условием нормировки
    omega = stationary_distribution(theta, 0.001)
    # м.о. числа требований в системах
    s = [[0] * L for _ in range(N + 1)]
    # м.о. длительности пребывания требований в системах
    u = [[0] * L for _ in range(N + 1)] 

    for Y in range(1, N + 1):
        for i in range(L):
            u[Y][i] = 1 / mu[i] * (s[Y-1][i] + 1)
        for i in range(L):
            summa = 0
            for j in range(L):
                summa += omega[j] * u[Y][j]
            s[Y][i] = omega[i] * u[Y][i] * Y / summa
            
    return s[N]



mu_0 = [0.1] * L # начальный вектор
delta = 0.1 # минимальный элемент изменяемого вектора
count = 0 # кол-во итераций
s = None # м.о. числа требований в системах

while True:
    s = get_s(L, N, mu_0, theta)
    if checking_for_identical_elements(s):
        break

    s_max = max(s)#выбираем максимальное значение из последовательности
    s_min = min(s)#выбираем минимальное значение из последовательности
    i = s.index(s_min)#получаем индекс минимального
    j = s.index(s_max)#получаем индекс максимального

    count += 1
    if count % 50 == 0:
        i = random.randint(0, len(s) - 1)
        j = random.randint(0, len(s) - 1)
        if s[i] > s[j]:
            i, j = j, i
        s_min = s[i]
        s_max = s[j]

    print('Перенос интенсивности обслуживания из', i+1 , 'в', j+1)
    print(s[i], '->', s[j])

    delta = min(mu_0[i], mu_0[j])
    gamma = random.random() * delta * (s_max - s_min) / s_max
    mu_0[i] -= gamma
    mu_0[j] += gamma

print('#' * 100)
print('mu при котором достигается равное м.о. длительности обслуживания всех систем:\n', mu_0)
print('м.о. длительности обслуживания систем:\n', s)


# ## Б)Необходимо подобрать такую маршрутную матрицу (при сохранениитопологии), чтобы коэффициенты использования приборов систем были одинаковы

# In[21]:


# Вариант № 10 
import numpy as np
import matplotlib.pyplot as plt

# функция для расчета стационарного решения
# решение системы omega*theta=omega с условием нормировки sum(omega)=1
def stationary_distribution(matrix, eps):
    L=matrix.shape[0]
    omega_0 = np.ones(matrix.shape[0])/(L)
    omega_current = omega_0 @ matrix
    while(np.linalg.norm(omega_0 - omega_current)>eps):
        omega_0=omega_current
        omega_current = omega_current @ matrix
    return omega_current


#функция для проверки равенства элементов
def checking_for_identical_elements(_list):
    delta = 0.01
    for i in range(1, len(_list)):
        if abs(_list[0] - _list[i]) > delta:
            return False
    return True

#функция для вычисления psy
def get_psy(L, N, mu, theta):
    # Решение уравнений omega*Theta = omega с условием нормировки
    omega = stationary_distribution(theta, 0.001)
    # м.о. числа требований в системах
    s = [[0] * L for _ in range(N + 1)]
    # м.о. длительности пребывания требований в системах
    u = [[0] * L for _ in range(N + 1)]
    # коэфф. использования систем
    psy = [0] * L
    # м.о. длительности ожидания требований в очереди системы
    w = [0] * L
    # м.о. числа требований, ожидающих обслуживание в очереди системы
    b = [0] * L
    # м.о. числа занятых приборов в системах
    h = [0] * L
    # интенсивность входящего потока требований в системы
    lambdas = [0] * L

    for Y in range(1, N + 1):
        for i in range(L):
            u[Y][i] = 1 / mu[i] * (s[Y-1][i] + 1)
        for i in range(L):
            summa = 0
            for j in range(L):
                summa += omega[j] * u[Y][j]
            s[Y][i] = omega[i] * u[Y][i] * Y / summa
            
    for i in range(L):
        w[i] = u[N][i] - (1 / mu[i])
        b[i] = s[N][i] * w[i] / u[N][i]
        h[i] = s[N][i] - b[i]
        lambdas[i] = h[i] * mu[i]
        psy[i] = lambdas[i] / mu[i]

    return psy

#начальные условия
L = 7
N = 25
mu = [1., 1., 1., 1., 1., 1., 1.]
theta = np.array([
    [.0, .6, .0, .4, .0, .0, .0],
    [.0, .0, .5, .0, .5, .0, .0],
    [.0, .0, .0, .3, .0, .7, .0],
    [.0, .0, .0, .0, .8, .0, .2],
    [.5, .0, .0, .0, .0, .5, .0],
    [.0, .3, .0, .0, .0, .0, .7],
    [.6, .0, .4, .0, .0, .0, .0]
])

def find_psy(L, N, mu, theta):
    for v1 in np.arange(.1, 1., .1):
        for v2 in np.arange(.1, 1., .1):
            for v3 in np.arange(.1, 1., .1):
                for v4 in np.arange(.1, 1., .1):
                    for v5 in np.arange(.1, 1., .1):
                        for v6 in np.arange(.1, 1., .1):
                            for v7 in np.arange(.1, 1., .1):
                                theta[0][1], theta[0][3] = v1, 1 - v1
                                theta[1][2], theta[1][4] = v2, 1 - v2
                                theta[2][3], theta[2][5] = v3, 1 - v3
                                theta[3][4], theta[3][6] = v4, 1 - v4
                                theta[4][0], theta[4][5] = v5, 1 - v5
                                theta[5][1], theta[5][6] = v6, 1 - v6
                                theta[6][0], theta[6][2] = v7, 1 - v7
                                psy = get_psy(L, N, mu, theta)
                                if checking_for_identical_elements(psy):
                                    print('#' * 100)
                                    return psy


ans_psy = find_psy(L, N, mu, theta)
print('psy:\n', ans_psy)
print()
print('theta:\n', theta)


# In[ ]:




