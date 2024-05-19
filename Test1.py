import numpy as np

class bra:
    def __init__(self,psi):
        assert isinstance(psi,ket)
        self.state=np.conj(psi.state.T)

class ket:
    def __init__(self):
        self.state=np.zeros((1,1),dtype=complex)

    @classmethod
    def from_given(cls,P):
        psi=cls()
        psi.state=np.array(P,dtype=complex)
        psi.state=np.reshape(psi.state,(len(P),1))
        return psi

    @classmethod
    def equally_weighted(cls,n):
        psi=cls()
        psi.state=np.ones((n,1),dtype=complex)
        return psi
    
    def step(self,U):
        self.state=U @ self.state

    def norm(self):
        return np.sqrt(sum(np.abs(self.state[i])**2 for i in range(self.state.shape[0])))
    
    def probability(self):
        return (np.abs(self.state)**2).ravel()
    def probability_walk(self):
        return (np.abs(self.state[0::2])**2+np.abs(np.roll(self.state,-1)[0::2])**2).ravel()
    

class superket(ket):
    @classmethod
    def two_kets(cls,psi1,psi2):
        psi=cls()
        psi.state=np.kron(psi1.state,psi2.state)
        return psi
    
    @classmethod
    def n_kets(cls,tab):
        psi=cls()
        psi.state=tab[0].state
        for i in range(1,len(tab)):
            psi.state=np.kron(psi.state,tab[i].state)
        return psi
    
    @staticmethod
    def categories(n):
        if n==0:
            return ['']
        else:
            return ['0'+el for el in superket.categories(n-1)]+['1'+el for el in superket.categories(n-1)]
    
if __name__=='__main__':
    P=np.array([i for i in range(4)])
    print(np.roll(P[::2],1)+P[::2])
