import math 
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import scipy.linalg as la



class FiniteDifference(object):
    def __init__(self, N,fvec,vexact, alpha, beta):
        self.b = np.pi                                 
        self.a = 0                                   
        self.N = N                                  
        self.H = (self.b -self.a) / (self.N+2)
        self.X = np.linspace(self.a, self.b, self.N+2).transpose()   
        self.Solution1 = np.array([])
        self.Exact  = np.array([])
        self.Error  = np.array([])
        self.fvec   = fvec
        self.vexact = vexact
        self.alpha = alpha
        self.beta = beta
        print(fvec.shape)

    def Matrix(self):    
        B     = (self.H**2)*self.fvec
        B[0]  = B[0] - self.alpha
        B[-1] = B[-1]-self.beta
        cc    = np.zeros(self.N)
        cc[0] = -2
        cc[1] = 1 
        u = la.solve_toeplitz(cc, B)
        u = np.hstack((self.alpha,u,self.beta))
        self.Error     = np.sqrt(self.H)*la.norm(u - self.vexact)
        self.Solution1 = u
        self.Exact     = self.vexact
        print(self.Error)
        

def ff(x):
    return -2*np.cos(x)*np.exp(-x)
def exx(x):
    return np.sin(x)*np.exp(-x)

N=2**5
N2 = 2**7
N3 =2**9
N4 = 2**11
N5 = 2**13

xx = np.linspace(0,np.pi,N+2)
x2= np.linspace(0,np.pi,N2+2)
x3= np.linspace(0,np.pi,N3+2)
x4= np.linspace(0,np.pi,N4+2)
x5= np.linspace(0,np.pi,N5+2)

                        
Testcase = FiniteDifference(N,ff(xx[1:-1]),exx(xx),0,0)
Testcase.Matrix()
Testcase4 = FiniteDifference(N2,ff(x2[1:-1]),exx(x2),0,0)
Testcase4.Matrix()
Testcase6 = FiniteDifference(N3,ff(x3[1:-1]),exx(x3),0,0)
Testcase6.Matrix()
Testcase8 = FiniteDifference(N4,ff(x4[1:-1]),exx(x4),0,0)
Testcase8.Matrix()
Testcase9 = FiniteDifference(N5,ff(x5[1:-1]),exx(x5),0,0)
Testcase9.Matrix()
plt.figure(1)
plt.plot(Testcase9.X, Testcase9.Solution1, label='approx')
plt.plot(Testcase9.X, Testcase9.Exact, label='exact')
plt.xlabel('x')
plt.ylabel('Solutions')
plt.title('Exact vs Approx')
plt.grid()

plt.figure(2)
deltaX = [1/Testcase.N,1/Testcase4.N,1/Testcase6.N,1/Testcase8.N,1/Testcase9.N] 
N = [N,N2,N3,N4,N5]
Error = [0.0209700211636,0.00544768957854,0.0013752776679,0.000344662620621,8.6218472083e-05]
plt.loglog(N,Error)
plt.legend(loc='upper right')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error vs N')
plt.grid()
plt.show()

plt.figure(3)
plt.loglog(deltaX,Error)
plt.legend(loc='upper right')
plt.xlabel('dx')
plt.ylabel('Error')
plt.title('Error vs dx')
plt.grid()
plt.show()





