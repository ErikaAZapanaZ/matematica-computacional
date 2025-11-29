import numpy as np
import numpy.polynomial as P


def intercambiafilas(A,fil_i,fil_j):
    A[[fil_i,fil_j],:] = A[[fil_j,fil_i],:]


def operacionFila(A,fil_m,fil_piv,factor):
    A[fil_m,:] = A[fil_m,:] - factor*A[fil_piv,:] 


def escalonaSimple(A):
      nfil = A.shape[0]
      ncol = A.shape[1]
      for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio) 

def escalonaConPiv(A):
      nfil = A.shape[0]
      ncol = A.shape[1]
      for j in range(0,nfil):
        imax = np.argmax(np.abs(A[j:nfil,j]))
        intercambiafilas(A,j+imax,j)
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio) 

def sustRegresiva(A,b):
    N = b.shape[0] 
    x = np.zeros((N,1))
    for i in range(N-1,-1,-1):
        x[i,0] = (b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
    return x 

def GaussElimSimple(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaSimple(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x

def GaussEimWhitPiv(A,b):
    Ab = np.append(A,b,axis=1) 
    escalonaConPiv(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x

def hilbert_matrix(n):
    A = np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            A[i-1,j-1]= 1/(i+j-1)
    return A

def LUdescomposition(A): # debe ser Matrx cuadrada
    nrows= A.shape[0]
    U=A.copy()
    L=np.eye(nrows,nrows,dtype=np.float64)

    for col in range(0,nrows-1):
        for  row in range(col+1,nrows):
            mult = U[row,col]/U[col,col]
            L[row,col]= mult
            operacionFila(U,row,col,mult)

    return(L,U)

def sutProgresiva(A,b):
     N = b.shape[0] 
     x = np.zeros((N,1))
     for i in range(0,N):
        x[i,0] = (b[i,0]-np.dot(A[i,0:i],x[0:i,0]))/A[i,i]
     return x

def SolveByLU(A,b):
    LU = LUdescomposition(A)
    L=LU[0]
    U=LU[1]

    Y = sutProgresiva(L,b)
    X = sustRegresiva(U,Y)

    return X

def interpLagrange(cx,cy):
    n= len(cx)
    p = P.Polynomial([0])
    for i in range(n):
        mascara = np.ones(n,dtype=bool)
        mascara[i] = False
        raices = cx[mascara]
        Laux = P.Polynomial.fromroots(raices)
        p = p + cy[i]*Laux/Laux(cx[i])
    return p