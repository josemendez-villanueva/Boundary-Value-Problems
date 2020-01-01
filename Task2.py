import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

class FiniteDifference(object):
    def __init__(self):
        self.b = 1                                          
        self.a = 0                                      
        self.N = 1000                                       
        self.H = (self.b -self.a) / self.N
        self.X = np.linspace(self.a, self.b, self.N+1).transpose()
        self.Solution = np.array([])    
    def Matrix(self):
        Main = -2*np.ones((self.N + 1 , 1)).ravel()          
        UpperLower = 1*np.ones((self.N, 1)).ravel()
        UpperLower2 = 1*np.ones((self.N, 1)).ravel()
        A = Main.shape[0]
        DIAGONAL = [Main, UpperLower,UpperLower2]
        BigA = sparse.diags(DIAGONAL, [0,-1,1], shape = (A,A)).toarray()
        BigA[0,0] = -2
        BigA[1,0] = 1
        BigA[self.N,self.N - 1] = 1
        BigA[self.N,self.N - 1] = 1
        
        VFunction = np.diagflat((800*(np.sin(self.X*np.pi)**2)))
        HamMatrix = ((-BigA)/(self.H**2)) + VFunction
        evalue, evector = np.linalg.eig(HamMatrix )
        
        z = np.argsort(evalue)
        z = z[0:6]
        Energies=(evalue[z])
            
        plt.figure(1, figsize=(12,10))
        for i in range(len(z)):
            plt.plot(self.X, ( evector[:, z[i]] /  np.sqrt(self.H) * 100 ) + Energies[i]  , lw=2, label="{} ".format(i))
            plt.xlabel('x', size=14)
            plt.ylabel('EigenVectors',size=14)
        plt.legend()
        plt.title('Wavefunctions',size=14)
        plt.show()
      
        plt.figure(2, figsize=(12,10))
        for i in range(len(z)):
            plt.plot(self.X, (((abs(evector[:, z[i]] * evector[:, z[i]])  / np.sqrt(self.H))) * 1000) + Energies[i]   , lw=2, label="{} ".format(i))
            plt.xlabel('x', size=14)
            plt.ylabel('EigenVectors',size=14)
        plt.legend()
        plt.title('Probability Density',size=14)
        plt.show()        

        VFunction2 = np.diagflat((700*(.5 - abs(self.X - .5))))
        HamMatrix2 = ((-BigA)/(self.H**2)) + VFunction2
        evalue2, evector2 = np.linalg.eig(HamMatrix2 )

        z2 = np.argsort(evalue2)
        z2 = z2[0:5]
        Energies2=(evalue2[z2])
            
        plt.figure(3, figsize=(12,10))
        for i in range(len(z2)):
            plt.plot(self.X, ((evector2[:, z2[i]]  / np.sqrt(self.H))*1000 ) + Energies2[i]  , lw=2, label="{} ".format(i))
            plt.xlabel('x', size=14)
            plt.ylabel('EigenVectors',size=14)
        plt.legend()
        plt.title('Wavefunctions 2',size=14)
        plt.show()

        plt.figure(4, figsize=(12,10))
        for i in range(len(z2)):
            plt.plot(self.X, (((evector2[:, z2[i]] * evector2[:, z2[i]])  / np.sqrt(self.H))*1000 )+ Energies2[i] , lw=2, label="{} ".format(i))
            plt.xlabel('x', size=14)
            plt.ylabel('EigenVectors',size=14)
        plt.legend()
        plt.title('Probability Density 2',size=14)
        plt.show()
    
        VFunction3 = np.diagflat(800*(np.sin(2*self.X*np.pi)**2))
        HamMatrix3 = ((-BigA)/(self.H**2)) + VFunction3
        evalue3, evector3 = np.linalg.eig(HamMatrix3 )

        z3 = np.argsort(evalue3)
        z3 = z3[0:7]
        
        Energies3=(evalue3[z3])
        
            
        plt.figure(5, figsize=(12,10))
        for i in range(len(z3)):
            plt.plot(self.X, ((evector3[:, z3[i]]  / np.sqrt(self.H))* 1000 ) + Energies3[i]   , lw=2, label="{} ".format(i))
            plt.xlabel('x', size=14)
            plt.ylabel('EigenVectors',size=14)
        plt.legend()
        plt.title('Wavefunctions 3',size=14)
        plt.show()

        plt.figure(6, figsize=(12,10))
        for i in range(len(z3)):
            plt.plot(self.X, (((evector3[:, z3[i]] * evector3[:, z3[i]])  / np.sqrt(self.H))* 1000 )+ Energies3[i]  , lw=2, label="{} ".format(i))
            plt.xlabel('x', size=14)
            plt.ylabel('EigenVectors',size=14)
        plt.legend()
        plt.title('Probability Density 3',size=14)
        plt.show()



Testcase = FiniteDifference()
Testcase.Matrix()
