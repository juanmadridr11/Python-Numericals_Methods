""" 
JUAN MANUEL MADRID RANGEL / PROGRAMACION 

Metodos numericos

                                                PARTE 1 / INTERPOLACION 

"""


# MODULO NUMPY : este modulo seriva para las distintas operaciones matemticas o expresiones.
# MODULO MATPLOTLIB.PYPLOT : este modulo servira para la graficacion de las curvas.

import numpy as np
import matplotlib.pyplot as plt

"""" 
Acontinuacion se definira la funcion para interpolarla. En python las funciones se definen como :  

def nombreDeLaFuncion(parametros)
    
    "en este espacio ira las operaciones matematicas o formulas 
    que el usuario quiera ingresar para definir la funcion."
    
    return 
    
El return es para indicarle al programa que te devuelva el valor de la operacion o en algunos caso de la variable     
    
"""

def interP(x,x0,y0,x1,y1,x2,y2,x3,y3):
    
    #formula de interpolacion cubica
    
    interCubica = y0 * ((x-x1)*(x-x2)*(x-x3))/((x0-x1)*(x0-x2)*(x0-x3)) + y1*((x-x0)*(x-x2)*(x-x3))/((x1-x0)*(x1-x2)*(x1-x3)) + y2*((x-x0)*(x-x1)*(x-x3))/((x2-x0)*(x2-x1)*(x2-x3)) + y3*((x-x0)*(x-x1)*(x-x2))/((x3-x0)*(x3-x1)*(x3-x2))
    
    #Condicion np.nan dada por el profesor

    if(x < 0.1 or x > 100):
        
        return np.nan
    
    else:
        
        return interCubica

"""
Aqui se extraen los datos en la carpeta especificada o el archivo del que se quieran extraer ayudades del metodo
de python "loadtxt()" 

"""

Datos = np.loadtxt("liq_water_density_coarse.txt",skiprows=2)

#Datos por columna 

"""
la syntaxis "[:,0]" indica que solo vamos a trabajar con las columnas de posicion en este 
caso 0 (depende el numero de posicion)

"""

x = Datos[:,0]

y = Datos[:,1]

temp = np.arange(0.1,100.+0.1,0.1)

"""
Ahora se llenaran unas matrices de zero para luego llenarlas con datos actualizados 

"""

xFinal = np.zeros_like(temp)

yFinal = np.zeros_like(temp)

#Gráficas

plt.figure()

plt.plot(x, y, 'r')

plt.xlabel('Temperatura')

plt.ylabel('Densidad')

"""
Ciclo donde miraremos el valor de temperatura que tomaremos 

"""

for i in range(0,temp.size):
    for j in range(0,x.size-3):
        
        
        if(temp[i] > x[j] and temp[i] < x[j+3]):
            
            xFinal[i] = temp[i]
            yFinal[i] = interP(temp[i],x[j],y[j],x[j+1],y[j+1],x[j+2],y[j+2],x[j+3],y[j+3])
            
        elif(temp[i]==x[j]):
            
            xFinal[i] = x[j]
            yFinal[i] = y[j]
            
        elif(temp[i]==100.):
            
            xFinal[i] = 100.
            yFinal[i] = 958.35
            
"""
En esta matriz de ceros es donde se guardaran los valores encontrados y actualizados

"""

datosFinales = np.zeros((xFinal.size,2))

"""
Matriz con datos encontrados 

"""
for i in range(0,xFinal.size): 
    
    datosFinales[i][0] = xFinal[i]
    
    datosFinales[i][1] = yFinal[i] 
    
"""
Tabla con los nuevos valores

"""
    
tablaFinal = np.savetxt("liq_water_density_fine", datosFinales, delimiter= '-') 

"""

                                           PARTE NUMERO 2
                                           
                                           METODO DE EULER  
                                           


"""



def odePenduloNoLineal(t, y):
   
    y1 = y[0]
    y2 = y[1]
    
    m = 3. 
    r = 5.  
    g = 10.
    b = 0.3
    

    dydt = np.array([y2, ( - b * y2 - m * g * np.sin(y1)) / (m * r)])
    
    return dydt

def odePenduloLineal(t,y):
    
    y1 = y[0]
    y2 = y[1]
    
   
    m = 3.
    r = 5.
    g = 10. 
    b = 0.3
    
    
    dydt = np.array([y2, ( - b * y2 - m * g * y1) / (m * r)])
    
    return dydt


dt = 0.01

""" CONDICIONES INICIALES """

tetaCaso1 = 15 # grados

tetaP = 0. # rad

# Tiempo final 

tf = 30.


tetaCaso2 = 45 #grados


tetaCaso3 = 75 #grados

# Vector tiempo

t = np.arange(0., tf + dt, dt)


y1 = np.zeros([2, t.size])  # theta = 15°

y2 = np.zeros([2, t.size])  # theta = 45°

y3 = np.zeros([2, t.size])  # theta = 75°

y4 = np.zeros([2, t.size])  # theta = 15°

y5 = np.zeros([2, t.size])  # theta = 45°

y6 = np.zeros([2, t.size])  # theta = 45°


"""
CONDICIONES INICIALES

"""
y1[0,0] = tetaCaso1 * np.pi/180 # theta = 15°

y1[1,0] = tetaP # thetap = 0

y2[0,0] = tetaCaso2* np.pi/180 # theta = 45°

y2[1,0] = tetaP # thetap = 0

y3[0,0] = tetaCaso3 * np.pi/180  # theta = 75°

y3[1,0] = tetaP # thetap = 0

y4[0,0] = tetaCaso1 # theta = 15°

y4[1,0] = tetaP # thetap = 0

y5[0,0] = tetaCaso2 # theta = 45°

y5[1,0] = tetaP # thetap = 0

y6[0,0] = tetaCaso3 # theta = 75°

y6[1,0] = tetaP # thetap = 0


for k in range(0, t.size - 1):
                                                #ECUACION NO LINEAL#
                                                     
    y1[:,k+1] = y1[:,k] + dt * odePenduloNoLineal(t[k], y1[:,k]) # con theta = 15°
    
    y2[:,k+1] = y2[:,k] + dt * odePenduloNoLineal(t[k], y2[:,k])  # con theta = 45°
    
    y3[:,k+1] = y3[:,k] + dt * odePenduloNoLineal(t[k], y3[:,k])  # con theta = 75°
    
                                                  #ECUACION LINEAL#
        
    
    y4[:,k+1] = y4[:,k] + dt * odePenduloLineal(t[k], y4[:,k])   # con theta = 15°
    
    y5[:,k+1] = y5[:,k] + dt * odePenduloLineal(t[k], y5[:,k])    # con theta = 45°
    
    y6[:,k+1] = y6[:,k] + dt * odePenduloLineal(t[k], y6[:,k])  #l con theta = 75°

    
                                             #SOLUCION ECUACION NO LINEAL#
        
tetaCaso12 = y1[0,:]  # para theta = 15°

tetaP1 = y1[1,:]   # para theta = 15° y thetap = 0

tetaCaso22 = y2[0,:]  # para theta = 45°

tetaP2 = y2[1,:]   # para theta = 45° y thetap = 0

tetaCaso32 = y3[0,:]  # para theta = 75°

tetaP3 = y3[1,:]  # para theta = 75° y thetap = 0 


                                             #SOLUCION ECUACION LINEAL#


tetaCaso42 = y4[0,:]  # para theta = 15°

tetaP4 = y4[1,:]   # para theta = 15° y thetap = 0

tetaCaso52 = y5[0,:]  # para theta = 45°

tetaP5 = y5[1,:]   # para theta = 45° y thetap = 0

tetaCaso62 = y6[0,:]  # para theta = 75°

tetaP6 = y6[1,:]   # para theta = 75° y thetap = 0


                                              #GRAFICAS T VS CASOSNN#
    
plt.figure()

plt.plot(t, tetaCaso12)

plt.plot(t, tetaCaso22)

plt.plot(t, tetaCaso32)

plt.plot(t, tetaCaso42)

plt.plot(t, tetaCaso52)

plt.plot(t, tetaCaso62)

plt.ylabel('GRADOS')

plt.xlabel('T(s)')

plt.figure()

plt.plot(t, tetaP1)

plt.plot(t, tetaP2)

plt.plot(t, tetaP3)

plt.plot(t, tetaP4)

plt.plot(t, tetaP5)

plt.plot(t, tetaP6)

plt.ylabel('Rad/s')

plt.xlabel('T(s)')




"""INTEGRACION NUMERICA PARTE 3"""
    
    
dx = t[1:] - t[:-1]

"""15° GRADOS"""

# INTEGRAL POR IZQUIERDA 

fxIzqui = (tetaCaso12[:-1] - tetaCaso42[:-1])**2
aIzqui = np.sum(fxIzqui * dx)

# INTEGRAL POR DERECHA 

fxDerecha = (tetaCaso12[1:] - tetaCaso42[1:])**2
aDrecha = np.sum(fxDerecha * dx)

#integral trapezoidal

aTrapezoidal1 = 0.5 * np.sum(dx*((fxIzqui + fxDerecha)))

print(" ")
print("         RESULTADO DE INTEGRALES  ")
print(" ")
print("Resultado igual a : ",aTrapezoidal1)


"""45° GRADOS"""

# INTEGRAL POR IZQUIERDA 

fxIzqui2 = (tetaCaso22[:-1] - tetaCaso52[:-1])**2

# INTEGRAL POR DERECHA

fxDerecha2 = (tetaCaso22[1:] - tetaCaso52[1:])**2

#integral trapezoidal

aTrapezoidal2 = (0.5 * np.sum(dx*((fxIzqui2 + fxDerecha2))))

print("Resultado igual a : ",aTrapezoidal2)


"""75° GRADOS"""


# INTEGRAL POR IZQUIERDA

fxIzqui3 = (tetaCaso32[:-1] - tetaCaso62[:-1])**2

fxDerecha3 = (tetaCaso32[1:] - tetaCaso62[1:])**2

aTrapezoidal3 = (0.5 * np.sum(dx*((fxIzqui2 + fxDerecha2))))

print("Resultado igual a : ",aTrapezoidal3)


