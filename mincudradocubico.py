import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import libreria as bib

# Define true model parameters
xa, xb= -5,5 # intervalo 
x=np.linspace(xa,xb, 100) # intervalo sobre el cual efectuamos el experimento
a, b, c, d=1, 2, 5, 4
y_exact=a+b*x+c*x**2+d*x**3

# simulate noisy data
m=30 # nuero de puntos en el intervalo -1,1
X=xa+(xb-xa)*np.random.rand(m) # intervalo [-5,5] puede varia depende de los datos de la linea 7
Y=a+b*X+c*X**2+d*X**3+50*np.random.randn(m) # el coeficientes de de 10 puede variar , mientras as grande mayor es la dispersion de los puntos 

# fit the data to the model using linear least square
'''
A=np.vstack([ X**0, X**1,X**2,X**3]) # see np.vander or np.vstack for alternative for
sol, r, rank, sv = la.lstsq(A.T, Y)
'''
At=np.array([ X**0, X**1,X**2,X**3])
auxMat=np.matmul(At,At.T)
np.reshape(Y,(m,1))
b=np.matmul(At,Y)
b= b.reshape(-1,1) #redimensiona b como vector columna ya que en elim gaus pivot b es un vector
sol=bib.GaussEimWhitPiv(auxMat,b)

y_fit=sol[0]+sol[1]*x+sol[2]*x**2+sol[3]*x**3
fig,ax = plt.subplots(figsize=(12,4))

ax.plot(X,Y,'go',alpha= 0.5, label='simulated data')
ax.plot(x,y_exact,'r',lw=2,label='true values $y=a+bx+cx^2+dx^3$')
ax.plot(x,y_fit,'b',lw=2,label='least square fit')
ax.set_xlabel(r'$x$', fontsize=18)
ax.set_ylabel(r'$y$', fontsize=18)
ax.legend(loc=2)
plt.show()