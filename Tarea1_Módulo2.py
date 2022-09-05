#MÃ©todo del descenso del Gradiente. 
import numpy as np
import random
import matplotlib.pyplot as plt

#Parámetros iniciales
x = []
y = []
for i in range(0, 60):
    x.append(i)
    y.append((i + 25) + random.uniform(0, 1) * 50)
    
m = 0
c = 0
learning_rate = 0.001
n = len(x)


# Algoritmo de gradiente descendiente
# Dm = Derivada Parcial con respecto a m
# Dc = Derivada Parcial con respecto a c
def gradiente(m, c):
    
    Dm = 0
    for i in range (0, n):
        Dm += (x[i] * ((m*x[i] + c) - y[i]))
    Dm = (learning_rate * Dm) / n
    
    Dc = 0
    for i in range(0,n):
        Dc += ((m*x[i] + c) - y[i])
    Dc = (learning_rate * Dc) / n
    
    return Dm, Dc
        

for i in range(0,10000):
    Dm, Dc = gradiente(m,c)
    m -= Dm
    c -= Dc

   
Error = 0
for i in range (0,n):
    Error += (y[i] - (m*x[i] + c))**2
    
Error = (1/n) * Error

print("Pendiente de la recta:", m)
print("Intercepción de y:", c)
print("Error", Error)

plt.scatter(x, y)
x2 = np.array(x)
y_calculada = c + m * x2
plt.plot(x2,y_calculada)
plt.show()