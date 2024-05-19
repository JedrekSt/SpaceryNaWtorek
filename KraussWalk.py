import numpy as np
from matplotlib import pyplot as plt 
from Spacery import *
from Test1 import *

class Decoherence(Walk):

    @staticmethod
    def Krauss1(alpha,dim,x0):
        assert alpha<=1 and alpha>=0, '0<=alpha<=1 '
        K1=np.zeros((dim,dim))
        K1[x0,x0]=1
        K1=np.kron(K1,np.identity(2))
        return np.sqrt(alpha)*K1,np.sqrt(np.identity(2*dim)-alpha*K1)
    
    @staticmethod
    def Evolution(A,U,Udag,K1,K2,pU):
        return pU*(U @ A @ Udag) + (1-pU)*(K1 @ A @ K1 + K2 @ A @ K2)
    
class Correlation:

    def __init__(self,dim):
        diag=np.exp(1j*2*np.pi/dim*np.array([i for i in range(dim)]))
        self.beta0=np.kron(np.diag(diag),np.identity(2))
        self.beta=np.conj(np.kron(np.diag(diag),np.identity(2)).T)

    def make_step(self,U,Udag,K1,K2,pU):
        self.beta=Decoherence.Evolution(self.beta,U,Udag,K1,K2,pU)

    def Correlation(self,psi0):
        return bra(psi0).state @ self.beta @ self.beta0 @ psi0.state
    
    @staticmethod
    def ft(data):
        return np.abs(np.fft.fft(data))

class DensityMat:
    
    def __init__(self):
        self.state=np.zeros((1,1))
    
    @classmethod
    def from_state(cls,psi):
        assert isinstance(psi,ket) or isinstance(psi,superket), 'psi must be an instance of ket'
        rho=cls()
        rho.state=np.kron(psi.state,bra(psi).state)
        return rho

    def probability(self):
        return np.diagonal(self.state)
    
    def probability_walk(self):
        prob=np.diagonal(self.state)
        return (prob[0::2]+np.roll(prob,-1)[0::2]).ravel()
    
    def step(self,U,Udag,K1,K2,pU):
        self.state=Decoherence.Evolution(self.state,U,Udag,K1,K2,pU)

    def std(self,p_walk):
        return np.sqrt(p_walk @ (np.arange(dim)**2)- (p_walk @ (np.arange(dim)))**2)

if __name__=='__main__':

    dim=71
    th=np.pi/4
    x0=dim//2
    sigma=2
    r=2
    v=2
    alpha=0.9
    pU=0.95
    n=[1,0,0]

    S=Walk.Step_shift(dim,r,v)
    C1=np.kron(np.identity(dim),Walk.Coin(th,n))
    C2=np.kron(np.identity(dim),Walk.Hadamard_Coin())
    K1,K2=Decoherence.Krauss1(alpha,dim,x0)
    U= S @ C1
    Udag=np.conj(U.T)
    Beta=Correlation(dim)

    Give=Walk.Give_Center(dim)
    GiveC=Walk.Give_Coin(1,1j)
    state=superket.two_kets(ket.from_given(Give),ket.from_given(GiveC))

    rho=DensityMat.from_state(state)
    prob=rho.probability_walk()

    Distribution=np.array([prob])
    Distr_beta=[]
    Distr_beta.append(Beta.Correlation(state)[0][0])

    std=[]
    std.append(rho.std(prob))

    time=300

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))

    for k in range(time):
        for i in range(2):
            for j in range(2):
                ax[i, j].cla()

        
        fig.suptitle('QRW', fontsize=16)
        fig.patch.set_facecolor('lightgrey')

        cax = ax[0, 0].imshow(np.real(Distribution), cmap='inferno', aspect='auto', origin='lower')
        ax[0, 0].set_xlabel('Position')
        ax[0, 0].set_ylabel('Time step')
        ax[0, 0].set_title('Probability distribution')
        bar = plt.colorbar(cax, ax=ax[0, 0], label='Probability')

        ax[0, 1].plot(range(dim), prob, marker='.')
        ax[0, 1].set_xlabel('Position')
        ax[0, 1].set_ylabel('Probability')
        ax[0, 1].set_title('Position probability distribution')
        

        Distr_beta.append(Beta.Correlation(state)[0][0])
        ft=Correlation.ft(np.array(Distr_beta))

        ax[1, 0].plot(range(len(ft)), ft, color='red')
        ax[1, 0].set_xlabel('Time')
        ax[1, 0].set_ylabel(r'$F(\langle \beta^{\dagger}(t)\beta \rangle)$')
        ax[1, 0].set_title('Correlation')
        
        ax[1, 1].plot(range(len(std)), std, color='red')
        ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_ylabel(r'$\sigma$')
        ax[1, 1].set_title('Standard deviation')

        Beta.make_step(U,Udag,K1,K2,pU)
        rho.step(U,Udag,K1,K2,pU)
        prob=rho.probability_walk()
        std.append(rho.std(prob))
        Distribution=np.concatenate((Distribution,np.array([prob])))

        plt.pause(0.01)
        bar.remove()

    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=2.0, w_pad=2.0, h_pad=5.0)
    plt.show()
