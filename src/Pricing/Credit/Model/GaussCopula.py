import numpy as np
import QuantLib as ql
from scipy.optimize import minimize,LinearConstraint
from scipy.stats import norm
from scipy.special import comb,factorial
from Pricing.Credit.Instruments import Credit_Index


def binomial_matrix(n:int,N:int) -> np.ndarray:

    res=np.zeros((n+1,N))
    for i in range(n+1):
        for j in range(N):
            res[i][j]=comb(n,i)*(j/N)**i*(1-j/N)**(n-i)
    
    return res

def E_min(Q:list,K:float,rho:float,nb_entities:int)->np.ndarray:

    if K==0:
        return 0
    if rho==1:
        rho-=1e-3

    points,weights=np.polynomial.hermite_e.hermegauss(10)
    nb_default=K*nb_entities

    res=np.zeros(len(Q))

    if (nb_default)%1!=0 :
        nb_default=int(nb_default)+1
    else:
        nb_default=int(nb_default)
    
    for i in range(len(Q)):
        cond_proba=norm.cdf( (norm.ppf(Q[i])-points*np.sqrt(rho))/np.sqrt(1-rho) )

        for j in range(nb_default):
            temp=comb(nb_entities,j)*(1-cond_proba)**(nb_entities-j)*cond_proba**j
            res[i]+=nb_default+ (j-nb_default)*np.sum(temp*weights)/np.sqrt(2*np.pi)
    return res

#Gauss
class Credit_Copula:

    def __init__(self,Curve,R=0.4):
        self.Curve,self.DF=Curve,Curve.Discount_Factor
        self.R=R

    def calibrate(self,instru:Credit_Index):
        self.attach,self.detach=instru.attach,instru.detach
        self.rho=instru.base_correls
        self.spread=instru.spread
        self.nb_entities=instru.nb_entities
        self.base_correl_interp=lambda x: np.interp(x,self.detach,self.rho,left=self.rho[0],right=self.rho[-1])
        cal=ql.Actual365Fixed()
        self.tgrid=np.array([cal.yearFraction(instru.schedule[0],d) for d in instru.schedule[1:]])

    def compute_default_proba(self,t):
        return 1-np.exp(-self.spread/(1-self.R)*t)
    
    def generate_default_time(self,rho:float,nb_simu:int):

        Z=np.random.normal(size=(nb_simu,self.nb_entities+1))
        Z_=np.sqrt(rho)*Z[:,0].reshape(nb_simu,1) + np.sqrt(1-rho)*Z[:,1:]
        U=norm.cdf(Z_)
        tau=np.array([-np.log(1-x)/(self.spread/(1-self.R)) for x in U])
        return tau
    
    def compute_emin_gaussian(self,K:float,rho:float) -> np.ndarray:

        if K==0:
            return 0
        if rho==1:
            rho-=1e-3

        points,weights=np.polynomial.hermite_e.hermegauss(100)
        nb_default=K*self.nb_entities/(1-self.R)

        res=np.zeros(len(self.tgrid))

        if (nb_default)%1!=0 :
            m=int(nb_default)+1
        else:
            m=int(nb_default)
        
        for i,t in enumerate(self.tgrid):
            cond_proba=norm.cdf((norm.ppf(self.compute_default_proba(t))-points*np.sqrt(rho))/np.sqrt(1-rho)) 
            for j in range(m):
                temp=comb(self.nb_entities,j)*(1-cond_proba)**(self.nb_entities-j)*cond_proba**j
                res[i]+=min(j*(1-self.R)/self.nb_entities,K)*np.sum(temp*weights)/np.sqrt(2*np.pi)
        
        return res

    def compute_survival_proba_tranche(self,K1:float,K2:float) -> float:

        rho1=self.base_correl_interp(K1)
        rho2=self.base_correl_interp(K2)

        E1=self.compute_emin_gaussian(K1,rho1)
        E2=self.compute_emin_gaussian(K2,rho2)

        return 1- (E2-E1)/(K2-K1)
    
    def solve_correlation(self,K1:float,K2:float) ->float:
        
        ZC=self.Curve.Discount_Factor(self.tgrid)
        Q_obj=self.compute_survival_proba_tranche(K1,K2)

        def compute_spread(ZC,Q):
            CL=sum(ZC[1:]*np.array([q2-q1 for q1,q2 in zip(Q[1:],Q)]))
            FL=sum(ZC*Q)

            return CL/FL

        spread_obj=compute_spread(ZC,Q_obj)

        def func_to_solve(x):
            Q=1-(self.compute_emin_gaussian(K2,x)-self.compute_emin_gaussian(K1,x))/(K2-K1)
            return (compute_spread(ZC,Q)-spread_obj)**2

        x0=self.base_correl_interp(K1)
        res=minimize(func_to_solve,x0,method='TNC',bounds=((1e-4,1),))
        
        return res.x[0]
