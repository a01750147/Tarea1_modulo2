#Método del descenso del Gradiente. 
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


# Regresión lineal con descencia del grandiente
def gradiente(m,m2,m3,c,learning_rate,n,Error_array,x,x2,x3,y):
    
    Dm = 0
    Dm2 = 0
    Dm3 = 0
    Dc = 0
    Error = 0
    
    for i in range (0, n):
        y1 = m*x[i]
        y2 = m2*x2[i]
        y3 = m3*x3[i]
        Dm += (x[i] * ((y1 + y2 + y3 + c) - y[i]))
        Dm2 += (x2[i] * ((y1 + y2+ y3 + c) - y[i]))
        Dm3 += (x3[i] * ((y1 + y2+ y3 + c) - y[i]))
        Dc += ((y1 + y2 + c) - y[i])
        Error += (y[i] - (y1  + y2 + y3 + c))**2
        
    Dm = (learning_rate * Dm) / n
    Dm2 = (learning_rate * Dm2) / n 
    Dm3 = (learning_rate * Dm3) / n 
    Dc = (learning_rate * Dc) / n    
    Error = Error/n
    Error_array.append(Error)

    return Dm, Dm2, Dm3, Dc, Error_array, Error
        

def modelo(m,m2,m3,c, learning_rate, iteraciones, x,x2,x3,y):
    n = len(x)
    #Para poder graficar el error
    iteraciones_array = []
    Error_array = []
    for i in range(iteraciones):
        iteraciones_array.append(i)
    
    for i in range(0,iteraciones):
        Dm, Dm2, Dm3, Dc, Error_array,Error = gradiente(m,m2,m3,c,learning_rate,n,Error_array,x,x2,x3,y)
        m -= Dm
        m2 -= Dm2
        m3 -= Dm3
        c -= Dc
    print("Pendiente de la recta (m1):", m)
    print("Pendiente de la recta (m2):", m2)
    print("Pendiente de la recta (m3):", m3)
    print("Error:", Error)
    
    plt.plot(iteraciones_array, Error_array)
    plt.show()
    return m,m2,m3

if __name__ == '__main__':
    #Entrenamiento 1
    print(' ** Modelo 1 ** ')
    train = pd.read_csv('Train.csv')
    x = np.array(train['1991'])
    x2 = np.array(train['1992'])
    x3 = np.array(train['1993'])
    y = np.array(train['1994'])
    m,m2,m3 = modelo(0,0,0,0,0.01,1000,x,x2,x3,y)
    
    
    #Prueba 1
    test = pd.read_csv('Test.csv')
    x = np.array(test['1991'])
    x2 = np.array(test['1992'])
    x3 = np.array(test['1993'])
    y_real = np.array(test['1994'])
    y_prediccion = []
    
    for i in range(len(x)):
        y = m*x[i] + m2*x2[i] + m3*x3[i]
        y_prediccion.append(y)
        
    parte1 = 0
    parte2 = 0
    for i in range(len(y_real)):
       parte1 += (y_prediccion[i] - np.mean(y_real))**2
       parte2 += (y_real[i] - np.mean(y_real))**2
    
    r2 = parte1 / parte2
    print("R2: ", r2)
       
    
    #Entrenamiento 2
    print(' ** Modelo 2 ** ')
    train = pd.read_csv('Train2.csv')
    x = np.array(train['1991'])
    x2 = np.array(train['1992'])
    x3 = np.array(train['1993'])
    y = np.array(train['1994'])
    m,m2,m3 = modelo(0,0,0,0,0.01,10000,x,x2,x3,y)
    
    
    #Prueba 2
    test = pd.read_csv('Test2.csv')
    x = np.array(test['1991'])
    x2 = np.array(test['1992'])
    x3 = np.array(test['1993'])
    y_real = np.array(test['1994'])
    y_prediccion = []
    
    for i in range(len(x)-1):
        y = m*x[i] + m2*x2[i] + m3*x3[i]
        y_prediccion.append(y)
        
    parte1 = 0
    parte2 = 0
    for i in range(len(y_real)-1):
       parte1 += (y_prediccion[i] - np.mean(y_real))**2
       parte2 += (y_real[i] - np.mean(y_real))**2
    
    r2 = parte1 / parte2
    print("R2: ", r2)
    
    #Predicciones
    print("** Predicciones **")
    predicciones = pd.read_csv('predicciones.csv')
    x = np.array(predicciones['1991'])
    x2 = np.array(predicciones['1992'])
    x3 = np.array(predicciones['1993'])
    y = np.array(predicciones['1994'])

    for i in range(len(x)-1):
        yp = m*x[i] + m2*x2[i] + m3*x3[i]
        print("Prediccion Y: ", yp.round(4),  "  Real y", y[i].round(4))



        
       



