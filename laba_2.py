#!/usr/bin/env python
# coding: utf-8

# # Задача 2.1

# # АНАЛИЗ СЕТЕЙ ОБСЛУЖИВАНИЯ МЕТОДОМ СВЕРТКИ

# In[14]:


# Вариант № 10 
import numpy as np

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

# функция для расчета нормализующих констант G(Q, L), Q+1 из-за начальных условий
# и рассмотрение варианта с 0 требований
def get_normalizing_constants(L, Q, x):
    g = [[0] * L for _ in range((Q + 1))]
    for i in range(Q + 1):
        g[i][0] = x[0]**i
    g[0] = [1] * L    
    
    for i in range(1, Q + 1):
        for j in range(1, L):
            g[i][j] = g[i][j-1] + x[j] * g[i-1][j]
            
    return g



# условия задачи
L = 8 # число система 
Q = 15 # число требований в закрытой системе
mu = np.array([1, 1.5, 0.8, 0.3, 0.5, 1.5,0.8,1]) # интенсивности обслуживания 
# маршурная матрица
theta = np.array([
    [0, 0, 0.5, 0.3, 0, 0, 0.2, 0],
    [0.8, 0, 0, 0, 0.1, 0.1, 0, 0],
    [0, 0, 0, 0.5, 0, 0, 0.5, 0],
    [0, 0.2, 0, 0, 0, 0.5, 0, 0.3],
    [0, 1, 0, 0, 0, 0,0,0],
    [0.5, 0, 0.3, 0, 0, 0, 0.2, 0],
    [0.4, 0, 0, 0, 0, 0, 0, 0.6],
    [0, 0.2, 0.5, 0, 0.3, 0,0,0],
])

# решение уравнений omega*theta = omega с условием нормировки
omega = stationary_distribution(theta, 0.001)
x = omega/mu

# вычисление нормализующей константы G(Q,L) и значения величин g(Y,L), Y = 1,..,Q; Z=1,2..,L 
g = get_g(L, Q, x)

# вер. что в системах m и более треб.
PM = [[0] * L for _ in range((Q + 1))]
# вер. что в системах ровно m треб.
Pm = [[0] * L for _ in range((Q + 1))]
# м.о. числа треб. в системах
s = [0] * L
# м.о. числа занятых приборов в системах
h = [0] * L
# интенсивности входного потока треб. в системах
lambdas = [0] * L
# м.о. длит. пребывания треб. в системах
u = [0] * L
# коэфф. использования систем
psy = [0] * L

# вычисление характеристик систем сети (по формулам 1.2)
for i in range(Q + 1):
    for j in range(L):
        PM[i][j] = x[j]**i * (g[Q-i][L-1] / g[Q][L-1])
        Pm[i][j] = (x[j]**i / g[Q][L-1]) * (g[Q-i][L-1] - x[j] * g[Q-i-1][L-1])

for i in range(L):
    for j in range(1, Q + 1):
        s[i] += x[i]**j * (g[Q-j][L-1] / g[Q][L-1])#м.о. числа требований в системе
    h[i] = x[i] * (g[Q-1][L-1] / g[Q][L-1])#м. о. числа занятых приборов
    lambdas[i] = h[i] * mu[i]#интенсивность входного потока требований в системе
    u[i] = s[i] / lambdas[i]#м. о. длительности пребывания требований в системе
    psy[i] = lambdas[i] / mu[i]#коэффициент использования системы
    
# вывод результатов  

print('\nСтационарные вероятности пребывания в системах m или более требований\n')
print(' ' * 40, 'Номер системы')
print('Кол.треб.', ' ' * 5, '1', ' ' * 7, '2', ' ' * 7, '3', ' ' * 7, '4', ' ' * 7,'5', ' ' * 7, '6', ' ' * 7, ' 7',' ' * 7, '8')
for i in range(Q + 1):
    print(i, end='\t| ')
    for j in range(L):
        print('{:07}'.format(round(PM[i][j], 5)), end=' | ')
    print('\n', '-' * 88)
        
print('#' * 100)   
        
print('\nСтационарные вероятности пребывания в системах ровно m требований\n')
print(' ' * 40, 'Номер системы')
print('Кол.треб.', ' ' * 5, '1', ' ' * 7, '2', ' ' * 7, '3', ' ' * 7, '4', ' ' * 7, '5', ' ' * 7, '6', ' ' * 7, ' 7', ' ' * 7, '8')
for i in range(Q + 1):
    print(i, end='\t| ')
    for j in range(L):
        print('{:07}'.format(round(Pm[i][j], 5)), end=' | ')
    print('\n', '-' * 88)
    
print('#' * 100)   
    
print('\nСтационарные характеристики сети\n')
print('Система', ' ' * 8, 's', ' ' * 16, 'u', ' ' * 12, 'lambdas', ' ' * 10, 'psy', sep='')
for i in range(L):
    print(i + 1, end='\t|')
    print(round(s[i], 5), round(u[i], 5), round(lambdas[i], 5), round(psy[i], 5), sep=' \t|', end=' \t|\n')
    print('-' * 73)


# # Задача 2.2

# ## РЕКУРСИВНЫЙ МЕТОД АНАЛИЗА СЕТЕЙ ОБСЛУЖИВАНИЯ

# In[21]:


# Вариант № 10 

import numpy as np

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

# условия задачи
L = 8 # число система 
Q = 100 # число требований в закрытой системе
mu = np.array([2.5, 2.5, 4, 3, 2.5, 3.5, 4, 3]) # интенсивности обслуживания 
# маршурная матрица
theta = np.array([
    [0,    0.4,    0.3,  0,  0,  0.2, 0.1, 0],
    [0,  0,    0,  0.4,    0.3,    0.3, 0, 0],
    [0.5,    0.1,  0,    0,  0,  0, 0.2, 0.2],
    [0.2,  0,  0.3,    0,    0.5,  0,  0,  0],
    [0,  0.1,    0,  0.3,  0,    0.6, 0, 0],
    [0,    0,    0.1,    0,    0,    0,0.5,0.4],
    [0.4, 0,  0, 0.1,    0.3,  0.1, 0, 0.1],
    [0,    0.5,    0,    0.2,    0,    0,0.3,0]
])

# решение уравнений omega*theta = omega с условием нормировки
omega = stationary_distribution(theta, 0.001)
x = omega/mu

# м.о. числа требований в системах
s = [[0] * L for _ in range(Q + 1)]
# м.о. длительности пребывания требований в системах
u = [[0] * L for _ in range(Q + 1)]
# м.о. длительности реакции сети обслуживания для систем
zita = np.zeros(L)
# коэффициенты использования систем
psy = np.zeros(L)
# м.о. длительности ожидания требований в очереди системы
w = np.zeros(L)
# м.о. числа требований, ожидающих обслуживание в очереди системы
b = np.zeros(L)
# м.о. числа занятых приборов в системах
h = np.zeros(L)
# интенсивность входящего потока требований в системы
lambdas = np.zeros(L)
# вероятность пребывания требований в системах
p = np.zeros(L)

#рекурсивным методом является следующее рекуррентное выражение
# вычисление м.о. длительности пребывания требований в системах и м.о. числа требований в системах
# по формулам (2.1) и (2.2), начально условие s_i|0=0
for Y in range(1, Q + 1):
    for i in range(L):
        u[Y][i] = (1 / mu[i]) * (s[Y-1][i] + 1)
    for i in range(L):
        summa = 0
        for j in range(L):
            summa += omega[j] * u[Y][j]
        s[Y][i] = omega[i] * u[Y][i] * Y / summa

# вычисление остальных характеристик сети
for i in range(L):
    w[i] = u[Q][i] - (1 / mu[i])
    b[i] = s[Q][i] * w[i] / u[Q][i]
    h[i] = s[Q][i] - b[i]
    lambdas[i] = h[i] * mu[i]
    psy[i] = lambdas[i] / mu[i]
    p[i] = s[Q][i] / Q
    summa = 0
    for j in range(L):
        if i != j:
            summa += omega[j] * u[Q][i]
    zita[i] = 1 / omega[i] * summa
    
    
#вывод результатов 
print('#' * 100)   
    
print('\nСтационарные характеристики сети\n')
print(' ' * 40, 'Характеристика')
print('Система', ' ' * 7, 'zita', ' ' * 12, 'omega', ' ' * 12, 'psy', ' ' * 14, 'u', ' ' * 15, 'w', sep='')

for i in range(L):
    print(i + 1, end='\t|')
    print(round(zita[i], 4), round(omega[i], 5), round(psy[i], 5), round(u[Q][i], 5), round(w[i], 5), sep=' \t|', end=' \t|\n')
    print('-' * 89)
    
print('#' * 100)    
    
print(' ' * 40, 'Характеристика')
print('Система', ' ' * 9, 'b', ' ' * 15, 's', ' ' * 12, 'lambdas', ' ' * 12, 'p', sep='')
for i in range(L):
    print(i + 1, end='\t|')
    print(round(b[i], 5), round(s[Q][i], 5), round(lambdas[i], 5), round(p[i], 5), sep=' \t|', end='  \t|\n')
    print('-' * 73)

