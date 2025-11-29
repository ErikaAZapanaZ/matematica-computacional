import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import libreria as bib

# Define true model parameters
x=np.linspace(-1,1,100) # intervalo sobre el cual efectuamos el experimento
a, b, c =1, 2, 150
y_exact=a+b*x+c*x**2

# simulate noisy data
m=20 # nuero de puntos en el intervalo -1,1
X=1-2*np.random.rand(m)
Y=a+b*X+c*X**2+10*np.random.randn(m) # el coeficientes de de 10 puede variar 
# fit the data to the model using linear least square
'''
A=np.vstack([ X**0, X**1,X**2]) # see np.vander or np.vstack for alternative for
sol, r, rank, sv = la.lstsq(A.T, Y)
'''


At=np.array([ X**0, X**1,X**2])
auxMat=np.matmul(At,At.T)
np.reshape(Y,(m,1))
b=np.matmul(At,Y)
b= b.reshape(-1,1) #redimensiona b como vector columna ya que en elim gaus pivot b es un vector
sol=bib.GaussEimWhitPiv(auxMat,b)

y_fit=sol[0]+sol[1]*x+sol[2]*x**2
fig,ax = plt.subplots(figsize=(12,4))

ax.plot(X,Y,'go',alpha= 0.5, label='simulated data')
ax.plot(x,y_exact,'r',lw=2,label='true values $y=1+2x+3x^2$')
ax.plot(x,y_fit,'b',lw=2,label='least square fit')
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$y$', fontsize=18)
ax.legend(loc=2)
plt.show()