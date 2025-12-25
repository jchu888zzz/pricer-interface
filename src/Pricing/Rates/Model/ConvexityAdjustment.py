import numpy as np
from scipy.interpolate import CubicSpline
import QuantLib as ql
import re

def convert_period(period:str) -> float:
    dic={'D':1/360,'W':7/360,'M':30/360,'Y':1}
    temp=re.split(r'(\D+)',period)
    return float(temp[0])*dic[temp[1]]

def G(x:float,n:int,q=1,delta=0.5) -> float:
    return  x/(1+x/q)**delta*1/(1-1/(1+x/q)**n)

def Gprime(x:float,n:int,q=1,delta=0.5)-> float:
    
    g1=x/(1+x/q)**delta    
    g1_prime=(1- x/q*delta/(1+x/q) )/(1+x/q)**delta
    
    g2=1/(1-1/(1+x/q)**n)
    g2_prime=-(n/q)*(1+x/q)**(-(n+1))/(1-(1+x/q)**(-n))**2
    
    return g1*g2_prime + g2*g1_prime

def GetAdjustment(Curve,calc_date:ql.Date,Instruments) -> dict:
        
    res=dict()
    for Term in set([x.Term for x in Instruments]):
        
        Instru=sorted([x for x in Instruments if x.Term==Term], key=lambda x: ql.Period(x.Expiry) )
        Expiries=[x.Expiry for x in Instru]
        Vols=np.array([x.vol for x in Instru])
        Vols=np.array([x.vol/100 if x.typequote== 'vol' else x.vol for x in Instru])
        T=convert_period(Term)

        cal=ql.Thirty360(ql.Thirty360.BondBasis)    
        Tgrid=[ cal.yearFraction(calc_date,calc_date +ql.Period(E)) for E in Expiries]
        L0=np.array([ sum(Curve.Discount_Factor(t+np.arange(0,T,1))) for t in Tgrid ])
        Forward=np.array([Curve.Forward_Swap_rate(t,T) for t in Tgrid ])
        G_=np.array([ Gprime(x,T) for x in Forward ])

        Adjusted= G_*L0*Vols**2*Tgrid
            
        res[Term]=CubicSpline(Tgrid,Adjusted)
            
    return res

def get_CMS_fwds(Curve,Curve_adj:dict,Term:str,tgrid:list,option='adjusted') -> np.ndarray:
    T=convert_period(Term)
    res=np.array([ Curve.Forward_Swap_rate(t,T) for t in tgrid ])
    if option=='unadjusted':
        return res
    else:
        return res+Curve_adj[Term](tgrid) 
