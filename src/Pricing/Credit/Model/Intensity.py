import QuantLib as ql
from scipy import optimize
import numpy as np
import pandas as pd
from Pricing.Utilities import  Dates,Functions
from Pricing.Credit import Instruments


def Calibration(curve,entity:Instruments.Credit_Single,cal=ql.Actual365Fixed(),display_details=False):
        
    interp_grid=[cal.yearFraction(entity.calc_date,entity.calc_date+ql.Period(t)) for t in entity.tenors ]
    spreads=np.interp(entity.tgrid,interp_grid,entity.quotes,left=entity.quotes[0],right=entity.quotes[-1])

    func_cl=lambda ZC,Q,x,h : ZC*Q*(1-np.exp(-x*h))
    func_fl=lambda ZC,Q,x,h : h*ZC*Q*np.exp(-x*h)
    func_acc=lambda ZC,Q,x,h: 0.5*h*ZC*Q*(1-np.exp(-x*h))

    Q=np.ones_like(entity.tgrid)

    values=np.zeros_like(entity.tgrid)
    tgrid=entity.tgrid
    
    cl,fl,acc=0,0,0
    
    ZC=curve.discount_factor(entity.schedule)
    for i,(t1,t2) in enumerate(zip(tgrid,tgrid[1:])):
        h=t2-t1
        func=lambda x:( (1-entity.R)*(cl + func_cl(ZC[i],Q[i-1],x,h)) 
                        -spreads[i]*(fl + func_fl(ZC[i],Q[i-1],x,h) +acc + func_acc(ZC[i],Q[i-1],x,h)) )

        values[i]=optimize.brentq(func,-0.5,0.5)

        cl+=func_cl(ZC[i],Q[i-1],values[i],h)
        fl+=func_fl(ZC[i],Q[i-1],values[i],h)
        acc+=func_acc(ZC[i],Q[i-1],values[i],h)

        Q[i]=Q[i-1]*np.exp(-values[i]*h)
    
    if display_details:
        print(pd.DataFrame({'Schedule':entity.schedule[:-1],'Default Proba':1-Q[:-1]}))

    return Intensity(curve,entity,values)

class Intensity:

    def __init__(self,curve,entity,values:np.ndarray):
        self.curve,self.DF=curve,curve.discount_factor
        self.values=values
        self.entity=entity

    def compute_survival_proba(self,t:float):
        standard_tgrid=self.entity.tgrid
        if t<=0:
            return 1
        integral=Functions.positive_integral_constant_by_part(self.values,standard_tgrid,t)
        return np.exp(-integral)
    
    def compute_default_proba(self,t:float):
        standard_tgrid=self.entity.tgrid
        if t<=0:
            return 0
        integral=Functions.positive_integral_constant_by_part(self.values,standard_tgrid,t)
        return 1-np.exp(-integral)

    def generate_jump(self,size):
        standard_tgrid=self.entity.tgrid
        seed=123
        rng=np.random.default_rng(int(seed))
        U=rng.uniform(0,1,size=size)

        res=[Functions.inverse_positive_integral_constant_by_part(self.values,standard_tgrid,-np.log(u),100)
            for u in U]

        return np.array(res)







