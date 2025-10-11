import numpy as np
import libreria as lib

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