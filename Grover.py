import numpy as np
from matplotlib import pyplot as plt
from Test1 import *

class Grover:
    @staticmethod
    def Hadamard(n):
        H0=1/np.sqrt(2)*np.array([
            [1,1],
            [1,-1]
        ])
        H=H0.copy()
        for _ in range(n-1):
            H=np.kron(H,H0)
        return H
    
    @staticmethod
    def Oracle(n,el):
        O=np.identity(2**n)
        num=sum(int(el[len(el)-1-i])*2**i for i in range(len(el)))
        O[num,num]=-1
        return O
    
    @staticmethod
    def Reflection(n,H):
        proj=np.zeros((2**n,2**n))
        proj[0,0]=1
        return H@(2*proj-np.identity(2**n))@H

    @staticmethod
    def Diffusion(n,el,H):
        return  Grover.Oracle(n,el)@Grover.Reflection(n,H)
    
if __name__=='__main__':

    def update(ax,categories,psik,iter):
        ax.cla()
        ax.bar(categories,psik.probability())
        ax.set_ylim(-0.1,1.3)
        ax.set_xlabel('state number')
        ax.set_ylabel('Probability')
        ax.set_title(f'iteration: {iter}')
        plt.pause(1)

    n=5
    el='100'

    H=Grover.Hadamard(n)
    D=Grover.Diffusion(n,el,H)

    Give=[1,0]
    Give/=np.sqrt(sum(np.abs(Give[i])**2 for i in range(len(Give))))
    psik=superket.n_kets([ket.from_given(Give) for _ in range(n)])

    iter=0
    fig,ax=plt.subplots()
    categories=superket.categories(n)

    update(ax,categories,psik,iter)
    psik.step(H)

    for _ in range(50):
        iter+=1
        update(ax,categories,psik,iter)
        psik.step(D)

    plt.show()
        