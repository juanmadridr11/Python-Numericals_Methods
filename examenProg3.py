""" 

parcial programacion

Juan manuel madrid rangel 

"""

# modulos 

import numpy as np 
import matplotlib.pyplot as plt 

#estimacion inicial  / punto principal 

x = 0.2

#valores de Re

d = 1e4
d2 = 1e5
d3 = 1e6

#valores particulares tomados por mi 

x2 = 0.1e-5
d1 = 1e2
x3 = 0

# funciones 

funcion_1 = lambda x: -2 * np.log10(2.51 / (d * np.sqrt(x))) - 1/np.sqrt(x)
funcion_2 = lambda x: -2 * np.log10(2.51 / (d2 * np.sqrt(x))) - 1/np.sqrt(x)
funcion_3 = lambda x: -2 * np.log10(2.51 / (d3 * np.sqrt(x))) - 1/np.sqrt(x)


#derivada de la funcion 

def devFuncion_1():
    
    x3 = x - ((d1-x2)/(funcion_1(d1)-funcion_1(x2))) * funcion_1(x2)
    
    return x3


# residual 

res = 1e-3

#numero de iteraciones maximas

nmax = 100


#ciclo para newthonrapshon

for k in range (0,nmax):
    
    x -= funcion_1(x)/devFuncion_1()
    
    print("Iteraciones :",k," // ",x-1)
    
    if (np.absolute(funcion_1(x)) < res):
        print("Iteraciones :",k+1," // ",x, "ES UNA BUENA ESTIMACION")
        break
    plt.plot(x,0,'o')

    k += k
    

#grafica de funcion con solucion de derivada

plt.Figure()
x1 = np.linspace(0,4,250)    
y1 = funcion_1(x1)
plt.grid()
plt.title("Funcion con solucion")
plt.plot(x1,y1)
plt.xlabel("valores")
plt.ylabel("Rango")

# grafica de datos obtenidos de Ff vs valore de Re

plt.figure()
d = np.linspace(1e4,1e5,30)
plt.title("Ff vs Re")
plt.grid()
plt.plot(d,funcion_1(x),'o')
plt.xlabel("Valores Re")
plt.ylabel("Valores Ff")



""" Punto dos """


# Parametros 

x0 = 1 #metros
y0 = 1 # metros 
xf = 10 # metros 
yf = 2 # metros 
T = 5 # segundos
g = 9.81 # m/s  


# Discretizacion 

N = 20
dT = T / N 
k1 = 1

# Vector coordenada x 

x = np.arange(0,T + dT , dT)
y = np.arange(0,T + dT , dT)


# Sistema

M = np.zeros([N+1,N+1])


#Frontera 0

M[0,0] = 1

# Nodos internos de dx/dt

for k in range (1,N):
    
    M[k,k-1] = 1
    M[k,k] = -52/25
    M[k,k+1] = 27/25
    
#Frontera 

M[-1,-2] = x0
M[-1,-1] = xf 

#Termino independiente

b = np.zeros([N+1,1])

b[N] = x0


# solucion de la ecuacion 

x1 = np.linalg.solve(M,b)

# Nodo interno de dy/dt

E = np.zeros([N+1,N+1])

E[0,0] = 1

for k in range (1,N):
    
    E[k,k-1] = 1
    E[k,k] = -52/25
    E[k,k+1] = 27/25
    
    
#Frontera 

E[-1,-2] = y0
E[-1,-1] = yf 

#Termino independiente

B = np.zeros([N+1,1])

for N in range (0,N):
    

    B[N] = -0.25



# solucion de la ecuacion 

y1 = np.linalg.solve(E,B)



# Graficas 

plt.figure()
plt.plot(T + x1,x,'r', linestyle ='--',linewidth = '3')
plt.grid()
plt.title('X VS T')
plt.xlabel('COORDENADA EN X')
plt.ylabel('TIEMPO')


plt.figure()
plt.plot(y,(T + y1),'k', linestyle ='-.',linewidth = '3')
plt.grid()
plt.title('Y VS T')
plt.xlabel('TIEMPO')
plt.ylabel('COORDENADA Y')

plt.figure()
plt.plot(x1,y1,'g', linestyle =':',linewidth = '3')
plt.grid()
plt.title('GRAFICA X VS Y')
plt.xlabel('DATOS X')
plt.ylabel('DATOS Y')


""" punto 3 """

# funcion derivada

def deriv(x1,x0,y1,y0):
    
    return (y1-y0)/(x1-x0)

# funcion Calculo de la derivada

def calculo(direccion,x1):
    
    dxdt = np.zeros([x1.size])
    
    for i in range(x1.size):
    
        if i == 0:
        
            dxdt[i] = deriv(x1[i+1], x1[i], direccion[i+1],direccion[i])
        
        elif i == (x1.size)-1 :
        
            dxdt[i] = deriv(x1[i], x1[i-1], direccion[i],direccion[i-1])
        
        else:
        
           dxdt[i] = deriv(x1[i+1], x1[i-1], direccion[i+1],direccion[i-1])
     
    return dxdt      

#funcion de prueba


def funcionp(x):
    return x**2 - 2*x - 5



arrayParaPrueba = np.linspace(-100,100,funcionp(x).size)
arrayParaPrueba2 = np.linspace(-100,100,funcionp(x).size)
derivadaDePrueba = calculo(funcionp(arrayParaPrueba),funcionp(arrayParaPrueba2))


plt.figure()
plt.title("Derivada de Funcion (arrayDePrueba)")
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(arrayParaPrueba,funcionp(arrayParaPrueba))
plt.plot(arrayParaPrueba,derivadaDePrueba)
plt.grid()


""" 

    AQUI COMPROBE QUE SI ES EFECTIVA LA FUNCION PARA DERIVACION NUMERICA POR
    LA GRAFICA "" DERIVADA DE FUNCION (arrayDePrueba) "" 
    

"""

# arrays de puntos arbitrarios


tiempo = np.linspace(0, 10,x1.size)
tiempo2 = np.linspace(0, 10,y1.size)
x5 = np.arange(0,x1.size ,1)
y5 = np.arange(0,y1.size,1)

# calculo de los arrays 

xp = calculo(x5, x1)
yp = calculo(y5, y1)

        

#Graficas

plt.figure()
plt.plot(xp,tiempo,'b',linewidth = '2') 
plt.title("DERIVADA X1")    
plt.xlabel('VELOCIDAD')
plt.ylabel('TIEMPO')
plt.grid()


plt.figure()
plt.plot(tiempo2,yp,'b',linewidth = '2') 
plt.title("DERIVADA Y1")    
plt.xlabel('TIEMPO')
plt.ylabel('VELOCIDAD')
plt.grid()


""" punto 4 """


# vectores

xp2 = np.array([0,
               xp[20]])
yp2 = np.array([0,
               yp[20]])
x2 = np.array([0,
               0.09])
y2 = np.array([0,
               -0.58])

#funciones


def funcion1(t,y):
    
    y2 = y[1]
    
    
    k = 1
    
    dydt = np.array([y2, -k * y2])
    
    return dydt
    
    

def funcion2(t,y):
    
    y2 = y[1]
    
    k = 3
    
    
    dydt = np.array([y2, -g -k * y2])
                    
    return dydt                

# RUNGE-KUTTA METODO 

def rungeKutta(fun,t,y0):
    
    y = np.zeros([y0.size, t.size])
    
    
    y[:,0] = y0
    
    
    for k in range(0,t.size - 1):
        
        h = t[k+1] - t[k]
        k1 = fun(t[k], y[:,k])
        k2 = fun(t[k] + h/2 , y[:,k] + h * k1/2)
        k3 = fun(t[k] + h/2 , y[:,k] + h * k2/2)
        k4 = fun(t[k] + h, y[:,k] + h *k3)
        y[:,k+1] = y[:,k] + h * (k1 + 2*k2 + 2*k3 + k4)/6
        
    return y    


#solucion
solucion1 = rungeKutta(funcion1, tiempo, xp2)
solucion2 = rungeKutta(funcion2, tiempo, yp2)
solucion3 = rungeKutta(funcion2, tiempo, x2)
solucion4 = rungeKutta(funcion2, tiempo, y2)
solucion10 = np.linspace(0,solucion1.size,x1.size)


# grafica para Funcion1 
plt.figure()
plt.grid()
plt.plot(x1,solucion10)
# grafica para Funcion2 
plt.figure()
plt.grid()
plt.plot(solucion10,y1)
# grafica para Funcion3
plt.figure()
plt.grid()
plt.plot(solucion10,xp)
# grafica para Funcion4 
plt.figure()
plt.grid()
plt.plot(solucion10,yp)
    

    


   

    
        
        
        
        
        
    


    





        
    
        
        
        
    


















    


  












    





    
    





 
    
   
    
    


 


























