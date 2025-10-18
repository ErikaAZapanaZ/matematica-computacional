import numpy as np
import libreria as lib
import time
#n=1000 # tamaño del sistea lineal
#A=np.random.rand(n,n)
#b=np.random.rand(n,1)
#start_time = time.perf_counter()
#sol = lib.GaussElimSimple(A,b)
#end_time = time.perf_counter()
#elapsed_time =end_time-start_time
#print(f"tiempo transcurrido:{elapsed_time:.4f} segundos")
A = np.random.randint(-5,10,size=(3,3))
print("jhgfd\n",A)
lib.intercambiaFilas(A,0,2)
print("jhgfd\n",A)
lib.operacionFila(A,2,0,2) 
print("jhgfd\n",A)
lib.escalonaSimple(A)
print(A)
lib.escalonaconPiv(A)
print(A)
b=np.random.randint(-5,10,size=(3,1))
print(b)
lib.sustRegresiva(A,b)  #resuelve un sistema escalonado
print(A)
sol = lib.GaussElimSimple(A,b)
print("la matriz original",A)
print("la solucion es  \n",sol)
print("comprobacion \n",A*sol)
print("el vector b es  \n",b)
residuo = A@sol-b
print("el residuo es:  \n",residuo)
print(" Norma del residuo:  \n",np.linalg.norm(residuo))

print("{:15s}{:25s}{:20}".format("n","cond","error"))
print("_"*50)
solutions=[]
for i in range (4,17):
    x = np.ones(i)
    H = lib.hilbert_matrix(i)
    b = H.dot(x)
    c = np.linalg.cond(H,2)
    xx = np.linalg.solve(H,b)
    err = np.linalg.norm(x-xx,np.inf)/np.linalg.norm(x,np.inf)
    solutions.append(xx)
    print("{:2d} {:20e} {:20e}".format(i,c,err))




