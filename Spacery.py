import numpy as np
from matplotlib import pyplot as plt
from Test1 import *

class Operators:
    sz=np.array([
        [1,0],
        [0,-1]
    ])
    sy=np.array([
        [0,-1j],
        [1j,0]
    ])
    sx=np.array([
        [0,1],
        [1,0]
    ])
    Id=np.array([
        [1,0],
        [0,1]
    ])
    rg=np.array([
        [1,0],
        [0,0]
    ])
    lf=np.array([
        [0,0],
        [0,1]
    ])

class Walk:
    @staticmethod
    def Hadamard_Coin():
        return 1/np.sqrt(2)*np.array([
            [1,1],
            [1,-1]
        ])
    @staticmethod
    def Coin(th,n):
        return np.array([
            [np.cos(th/2)-1j*n[2]*np.sin(th/2),-(n[1]+1j*n[0])*np.sin(th/2)],
            [(n[1]-1j*n[0])*np.sin(th/2),np.cos(th/2)+1j*n[2]*np.sin(th/2)]
        ])
    
    @staticmethod
    def Step(n):
        Sup=np.zeros((n,n))
        Sdown=np.zeros((n,n))
        for i in range(n):
            Sup[(i+1)%n,i]=1
            Sdown[i,(i+1)%n]=1
        return np.kron(Sup,Operators.rg)+np.kron(Sdown,Operators.lf)
    
    @staticmethod
    def Step_shift(n,r,v):
        S=np.identity(n)
        perm=[i for i in range(n)]
        S1=S[np.roll(perm,r-v),:]
        S2=S[np.roll(perm,-r-v),:]
        return np.kron(S1,Operators.rg)+np.kron(S2,Operators.lf)
    
    @staticmethod
    def Give_Gauss(x0,sigma,dim):
        Give=np.array([np.exp(-(i-x0)**2/(sigma**2)) for i in range(dim)])
        Give/=np.sqrt(sum(np.abs(Give[i])**2 for i in range(len(Give))))
        return Give
    
    @staticmethod
    def Give_Center(dim):
        Give=np.zeros(dim)
        Give[dim//2]=1
        return Give
    
    @staticmethod
    def Give_Coin(a,b):
        GiveC=[a,b]
        GiveC/=np.sqrt(sum(np.abs(GiveC[i])**2 for i in range(len(GiveC))))
        return GiveC

    
if __name__=='__main__':
    dim=71
    th=np.pi/4
    r=1
    v=5
    n=[0,1,0]

    S=Walk.Step_shift(dim,r,v)
    C1=np.kron(np.identity(dim),Walk.Coin(th,n))
    C2=np.kron(np.identity(dim),Walk.Hadamard_Coin())

    sigma=2
    x0=dim//2
    Give=np.array([np.exp(-(i-x0)**2/(sigma**2)) for i in range(dim)])
    Give/=np.sqrt(sum(np.abs(Give[i])**2 for i in range(len(Give))))
    
    
    GiveC=[1,-1j]
    GiveC/=np.sqrt(sum(np.abs(GiveC[i])**2 for i in range(len(GiveC))))

    state=superket.two_kets(ket.from_given(Give),ket.from_given(GiveC))
    prob=state.probability_walk()
    Distribution=np.array([prob])

    fig, ax = plt.subplots(1,2)

    time=500
    for k in range(time):

        ax[0].cla()
        ax[1].cla()
        plt.title('Quantum Walk Probability Distribution')
        cax = ax[0].imshow(Distribution, cmap='inferno', aspect='auto', origin='lower')
        bar=plt.colorbar(cax, label='Probability')
        ax[0].set_xlabel('Position')
        ax[0].set_ylabel('Time step')
        pax = ax[1].plot(range(dim),prob,marker='.')
        ax[1].set_xlabel('Position')
        ax[1].set_ylabel('Probabbility')
        plt.fill_between(range(dim),prob, alpha=0.3)

        
        plt.pause(0.01)
        if k!=time-1:
            bar.remove()

        state.step(C2)
        state.step(S)
        prob=state.probability_walk()
        Distribution=np.concatenate((Distribution,np.array([prob])))
    
    plt.show()

    





