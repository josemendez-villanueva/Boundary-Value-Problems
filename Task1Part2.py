
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


class FiniteDifference(object):
    def __init__(self):
        self.b = 10                                          #Final time
        self.a = 0                                          #Start time
        self.N = 999                                        #Number of steps wanted to create the equally spaced grid
        self.H = (self.b -self.a) / self.N
        self.X = np.linspace(self.a, self.b, self.N+1).transpose()  #Transpose to get in correct format for when it runs code
        self.Solution1 = np.array([])
        self.Solution2 = np.array([]) 
        self.E = 1.9e11
    def Matrix(self):
        B = np.zeros((self.N+1, 1)).ravel()                      
        B[1:self.N] = -50000*(self.H**2 )  #To convert kN/m to N/m: You multiply by 1000 as 1000N = 1kN
        Main = -2*np.ones((self.N + 1, 1)).ravel()          
        UpperLower = 1*np.ones((self.N, 1)).ravel()
        UpperLower2 = 1*np.ones((self.N, 1)).ravel()
        A = Main.shape[0]
        DIAGONAL = [Main, UpperLower,UpperLower2]
        BigA = sparse.diags(DIAGONAL, [0,-1,1], shape = (A,A)).toarray()
        BigA[0,0] = -2
        BigA[1,0] = 1
        BigA[self.N,self.N - 1] = 1
        BigA[self.N,self.N - 1] = 1
        self.Solution1 = np.append(self.Solution1, np.linalg.solve(BigA, B))
        C = np.zeros((self.N+1, 1)).ravel()
        C[1:self.N] = self.Solution1[1:self.N]/(.001*(self.E * (3-2*(np.cos((self.X[1:self.N]*np.pi))**12)/self.b))) * (self.H**2 )
        self.Solution2 = np.append(self.Solution2, np.linalg.solve(BigA, C))  
        print(min(self.Solution2) )
        print(self.Solution1)
    
 

Testcase = FiniteDifference()
Testcase.Matrix()
plt.figure(1)
plt.plot(Testcase.X,Testcase.Solution2)
plt.plot([3,7], [min(Testcase.Solution2), min(Testcase.Solution2)])
plt.xlabel('Length')
plt.ylabel('Defelction')
plt.title('Deflection @ Midpoint')
plt.show()
#print(Testcase.Solution2)

