import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq,least_squares
from scipy.interpolate import CubicSpline
import QuantLib as ql
from typing import Union
from scipy.stats import pearsonr 

from Pricing.Utilities.Dates import compute_target_schedule
from Pricing.Utilities.decorators import timer 
import Pricing.Rates.Model.ConvexityAdjustment as CA
from Pricing.Utilities.InputConverter import convert_period
import Pricing.Rates.Instruments as Rate_Instruments
import Pricing.Utilities.Functions as Functions
from Pricing.Curves import Classic


def get_model(dic_df:dict,currency:str,calc_date) -> dict:
    calc_date=ql.Date.todaysDate()
    curve,risky_curve=Classic.get_curves(calc_date,dic_df,currency)

    swaptions_rate=Rate_Instruments.get_swaptions(dic_df['swaption'],
                                                curve,calc_date,currency)
    swaptions_rate=[x for x in swaptions_rate if x.striketype=='ATM']
    model=Calibration(curve,swaptions_rate)
    return {'risky_curve':risky_curve,
            'curve':curve,
            'model':model,
            'calc_date':calc_date}


def Put(vol:float,K:float,ZC1:float,ZC2:float)-> float:
    
    d1=(1/vol)*np.log(ZC1*K/ZC2)+ 0.5*vol
    d2=d1-vol
    return K*ZC1*norm.cdf(d1) - ZC2*norm.cdf(d2)

@timer
def Calibration(Curve,instru_calib:list[Rate_Instruments.Swaption | Rate_Instruments.Cap]):
    Caps=[x for x in instru_calib if isinstance(x,Rate_Instruments.Cap)]
    Swaptions=[x for x in instru_calib if isinstance(x,Rate_Instruments.Swaption) ]
    def error_function(param:tuple[float]):
        model=HW(Curve,param)
        res=[]
        for item in Swaptions:
            res.append((item.price-model.Swaption_price(item))/1e4)
        for item in Caps:
            res.append((item.price-model.Cap_price(item))/1e4)

        return res

    optimizer=least_squares(error_function,x0=(0.02,0.02),bounds=([1e-5,1e-3],[0.5,np.sqrt(0.1)]))

    return HW(Curve,optimizer.x)


def select_rates(rates:np.ndarray,grid_simu:list,tgrid:list,depth=1):
    if depth==1:
        idx=[Functions.find_idx(grid_simu,t) for t in tgrid]
        return rates[:,idx].T
    else:
        tgrid1=np.insert(tgrid,0,0)
        res=np.zeros((len(tgrid),depth,rates.shape[0]))
        for i,(t1,t2) in enumerate(zip(tgrid1,tgrid1[1:])):
            subgrid=np.linspace(t1,t2,depth)
            sub_idx=[Functions.find_idx(grid_simu,t) for t in subgrid]
            res[i]=(rates[:,sub_idx].T)
        return res

class HW :
    
    def __init__(self,Curve,param_=(0.02,0.02)):
        #Forward and DF are interpolation function
        self.Curve=Curve
        self.spot=Curve.spot
        self.DF=CubicSpline(Curve.timegrid,Curve.value)
        self.a,self.sigma=param_[0],param_[1]
        
    def __repr__(self):
        return f'HW(a:{self.a},sigma:{self.sigma})'
    
    #Smoother instantaneous forward rate
    def instantaneous_f(self,t,h=0.1):
        res=-(np.log(self.DF(t+h))-np.log(self.DF(t)))/h
        return res
    
    # def instantaneous_f(self,t:float,T:float,h=1e-3):
    #     res=-(np.log(self.DF(T+h))-np.log(self.DF(t)))/(T-t+h)
    #     return res 
    
    def B_term(self,delta:Union[np.ndarray,float],factor=1) -> Union[np.ndarray,float]:
        if (self.a==0):
            return delta
        else:
            a=factor*self.a
            return (1-np.exp(-a*delta))/a

    def affine_term(self,t:float,T:Union[np.ndarray,float]) ->tuple[np.ndarray]:
        forward=self.instantaneous_f(t)
        B=self.B_term(T-t)
        A= self.DF(T)/self.DF(t)*np.exp(B*forward - 0.5*(B**2)*self.sigma**2*self.B_term(t,factor=2))
        return (A,B)
    
    def compute_discount_factor_from_rates(self,rate:list,t:float,T:Union[np.ndarray,float])->np.ndarray:
        A,B=self.affine_term(t,T)
        return np.array([A*np.exp(-r*B) for r in rate])
    
    def var_(self,t:Union[np.ndarray,float]) -> Union[np.ndarray,float]:
         if self.a==0:
            return 0.5*self.sigma**2*t
         else:
            return 0.5*self.sigma**2*(1-np.exp(-2*self.a*t))/self.a
        
    def alpha_T(self,t:float,T:float) ->float:
        if self.a==0:
            term1=0.5*self.sigma**2*t
            term2=self.sigma**2*t/(T-0.5*t)
        else:
            term1=0.5*(self.sigma/self.a)**2*(1-np.exp(-self.a*t))**2
            term2=(self.sigma/self.a)**2*((1-np.exp(-self.a*t)) - np.exp(-self.a*T)*np.sinh(self.a*t))

        return self.instantaneous_f(t) + term1 -term2
    
    def Cap_price(self,item:Rate_Instruments.Cap)->float:
        
        K_prime=(1+item.K*item.delta)
        ZC_ratio=[x/y for x,y in zip(item.ZC,item.ZC[1:])]
        
        vol_i=self.B_term(item.delta)*self.sigma*np.sqrt(self.B_term(item.timegrid[:-1],factor=2))
        d1=(1/vol_i)*np.log(ZC_ratio/K_prime) + 0.5*vol_i
        d2=d1-vol_i
    
        return np.sum(item.ZC[:-1]*norm.cdf(d1)-K_prime*item.ZC[1:]*norm.cdf(d2))*10**4
    
    def Swaption_price(self,item:Rate_Instruments.Swaption)->float:
        c=item.delta.copy()*item.K
        c[-1]+=1
        
        A_,B_=self.affine_term(item.timegrid[0],item.timegrid[1:])

        func_to_solve= lambda x : np.sum(c*A_*np.exp(-B_*x)) -1 
        r_star=brentq(func_to_solve,-0.8,0.8)
        X=A_*np.exp(-B_*r_star)

        vol_=B_*self.sigma*np.sqrt(self.B_term(item.timegrid[0],factor=2))
                 
        return np.sum(c*Put(vol_,X,item.ZC[0],item.ZC[1:]))*10**4

    def get_CMS_adjustment(self,Curve,calc_date:ql.Date,Instruments:list):
        self.cms_adj=CA.GetAdjustment(Curve,calc_date,Instruments)
    
    #52 = weeks,360 days
    def rate_simulation(self,T:float,grid:list,Nb_simu:int,rng:np.random._generator.Generator):
        
        alpha,Var=[self.alpha_T(t,T) for t in grid ],[self.var_(t) for t in grid]
        res=np.zeros((Nb_simu,len(grid)))
        # initial_t=0.25
        # curve=self.Curve
        # res[:,0]=curve.L(0,initial_t)
        initial_t=T/4
        res[:,0]=-(np.log(self.DF(initial_t))-np.log(self.DF(0)))/initial_t
        for i in range(1,len(grid)):
            h=grid[i]-grid[i-1]

            res[:,i]=( res[:,i-1]*np.exp(-self.a*h) + alpha[i] - alpha[i-1]*np.exp(-self.a*h) 
                      + np.sqrt(Var[i]-Var[i-1]*np.exp(-2*self.a*h))*rng.standard_normal(size=Nb_simu) )
        
        return res
    
    def generate_rates(self,calc_date:ql.Date,maturity_date:ql.Date,
                       cal=ql.Thirty360(ql.Thirty360.BondBasis),
                       Nbsimu=10000,seed=123) -> dict:
        #seed=int(sum(Contract.fixgrid))
        rng=np.random.default_rng(int(seed))
        T=cal.yearFraction(calc_date,maturity_date)
        schedule=compute_target_schedule(calc_date,maturity_date,ql.Period('1D'))
        grid=np.array([cal.yearFraction(calc_date,x) for x in schedule[1:] ])
        
        rates=self.rate_simulation(T,grid,Nbsimu,rng)

        return {'rates':rates,'schedule':schedule,'grid':grid}

    # def rate_simulation(self,T:float,grid:list,Nb_simu:int,rng:np.random._generator.Generator):
        
    #     def func_alpha(t):
    #         return self.instantaneous_f(t) + 0.5*(self.sigma/self.a)**2*(1-np.exp(-self.a*t))**2
        
    #     def func_M(s,t):
    #         return (self.sigma/self.a)**2*((1-np.exp(-self.a*(t-s))) - 0.5*np.exp(-self.a*T)*(np.exp(self.a*t)-np.exp(-self.a*(t-2*s))))

    #     alpha=[func_alpha(t) for t in grid ]
    #     M_=[func_M(0,grid[0])]+[func_M(s,t) for s,t in zip(grid,grid[1:])]
    #     Var=[self.var_(t) for t in grid]
    #     Res=np.zeros((Nb_simu,len(grid)))
    #     h=1e-3
    #     Res[:,0]=-(np.log(self.DF(h))-np.log(self.DF(0)))/h
    #     X=np.zeros((Nb_simu,len(grid)))
    #     for i in range(1,len(grid)):
    #         h=grid[i]-grid[i-1]
    #         #print(-M_[i]+M_[i-1]*np.exp(-self.a*h))
    #         X[:,i]=X[:,i-1]*np.exp(-self.a*h) - M_[i-1] + np.sqrt(Var[i]-Var[i-1]*np.exp(-2*self.a*h))*rng.standard_normal(size=Nb_simu) 
    #         Res[:,i]=X[:,i] + alpha[i] 

    #     return Res
    
    def compute_deposit_from_rates(self,rates:np.ndarray,t:float,DepositTerm:str) -> np.ndarray:
        h=convert_period(DepositTerm)
        P=self.compute_discount_factor_from_rates(rates,t,t+h)

        return (1-P)/(P*h)
    
    def compute_cms_from_rates(self,rates:np.ndarray,t:float,Term:str,h=1) -> np.ndarray:
        T=convert_period(Term)
        grid=t+np.arange(0,T,h)
        P=self.compute_discount_factor_from_rates(rates,t,grid)
        res=np.array([(x[0]-x[-1])/np.sum(x[1:]) for x in P]) 
        
        return res
    
    def compute_single_undl_from_rates(self,data_rates:dict,fixgrid:list,undl1:str,include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""
        cur1,rate_type1,tenor1=undl1.split()
        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)

        if rate_type1=="CMS":
            undl=np.array([self.compute_cms_from_rates(rates[i],fixgrid[i],tenor1) for i in range(len(fixgrid))])
        elif rate_type1=="Euribor":
            undl=np.array([self.compute_deposit_from_rates(rates[i],fixgrid[i],tenor1) for i in range(len(fixgrid))])
        else:
            raise ValueError(f"{rate_type1} not implemented")
        if not include_rates:
            return {'undl':undl}
        else:
            return {'undl':undl,'rates':rates}
    
    def compute_single_undl_from_rates_with_depth(self,data_rates:dict,fixgrid:list,undl1:str,fixing_depth:int,
                                                  include_rates=True)->np.ndarray:
        """ result shape (len(fixgrid),fixing_depth,nb simu)"""
        cur1,rate_type1,tenor1=undl1.split()
        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,fixing_depth)
        res=np.zeros_like(rates)
        
        tgrid=np.insert(fixgrid,0,0)
        for i,t in enumerate(tgrid[:-1]):
            sub_fixgrid=np.linspace(t,tgrid[i+1],fixing_depth) # nbweeks 52
            if rate_type1=="CMS":
                res[i]=np.array([ self.compute_cms_from_rates(rates[i][j],sub_fixgrid[j],tenor1) 
                             for j in range(len(sub_fixgrid))])
            elif rate_type1=="Euribor":
                res[i]=np.array([ self.compute_deposit_from_rates(rates[i][j],sub_fixgrid[j],tenor1) 
                             for j in range(len(sub_fixgrid))])
        
        if not include_rates:
            return {'undl':res}
        else:
            return {'undl':res,'rates':rates[:,-1,:]}
    
    def compute_fix_rate_basis_func_from_rates(self,data_rates:dict,
                                               fixgrid:list[ql.Date],paygrid:list[ql.Date],
                                               call_idxs:list[ql.Date],
                                               deg=3,max_correlation=0.85,
                                               include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""
        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)
        undl=[None]*len(call_idxs)
        for i,idx in enumerate(call_idxs):
            swapgrid=np.arange(fixgrid[idx],paygrid[-1],0.5)
            delta=np.array([x-y for x,y in zip(swapgrid[1:],swapgrid)])
            Pt_T=self.compute_discount_factor_from_rates(rates[idx],fixgrid[idx],swapgrid)
            swap_values=np.array([(P[0]-P[-1])/np.sum(delta*P[1:]) for P in Pt_T ])
            undl[i]=swap_values.copy()
            # for j in range(2,deg+1):
            #     undl[i]=np.c_[undl[i],swap_values**j]
            # stop_grid=np.array([paygrid[j] for j in call_idxs[i+1:]+[len(paygrid)-1]])
            # bond_values=self.compute_discount_factor_from_rates(rates[i],fixgrid[i],stop_grid)  
            # bond_values=self.compute_discount_factor_from_rates(rates[i],fixgrid[i],paygrid[idx+1:])
            # temp=[]
            # for b in bond_values.T :
            #     if abs(pearsonr(b,swap_values).statistic) <=max_correlation:
            #         temp.append(b)
            # if temp:        
            #     undl[i]=np.c_[np.array(temp).T,undl[i]] 

            undl[i]=(undl[i] - np.mean(undl[i],axis=0))/np.std(undl[i],axis=0)

        if not include_rates:
            return {'undl':undl}
        else:
            return {'undl':undl,'rates':rates}
        
    def compute_swaption_from_rates(self,data_rates:dict,
                                               fixgrid:list[ql.Date],paygrid:list[ql.Date],
                                               call_idxs:list[ql.Date],K:float,
                                                side='sell',
                                               include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""
        
        if side=='buy':
            compute_price= lambda DF,K,x,Pt_T,delta: DF*np.maximum(x-K,0)*np.sum(delta*Pt_T[:,1:],axis=1)
        elif side=='sell':
            compute_price= lambda DF,K,x,Pt_T,delta: DF*np.maximum(K-x,0)*np.sum(delta*Pt_T[:,1:],axis=1)
        else: 
            raise ValueError(' Invalid input as side {side}')

        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)
        undl=[None]*len(call_idxs)
        for i,idx in enumerate(call_idxs):
            swapgrid=np.arange(fixgrid[idx],paygrid[-1]+1,0.5)
            delta=np.array([x-y for x,y in zip(swapgrid[1:],swapgrid)])
            Pt_T=self.compute_discount_factor_from_rates(rates[idx],fixgrid[idx],swapgrid)
            swap_values=np.array([(P[0]-P[-1])/np.sum(delta*P[1:]) for P in Pt_T ])
            
            undl[i]=compute_price(self.DF(fixgrid[idx]),K,swap_values,Pt_T,delta)
            #undl[i]=(undl[i] - np.mean(undl[i],axis=0))/np.std(undl[i],axis=0)

        if include_rates:
            return undl,rates 
        else:
            return undl
        
    def compute_prep_for_swaption_from_rates(self,data_rates:dict,
                                               fixgrid:list[float],T:float,
                                               call_idxs:list[int],
                                               include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""

        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)
        res_delta=[None]*len(call_idxs)
        res_Pt_T=[None]*len(call_idxs)
        res_swap=[None]*len(call_idxs)
        res_DF=[None]*len(call_idxs)
        for i,idx in enumerate(call_idxs):
            swapgrid=np.arange(fixgrid[idx],T+1,0.5)
            delta=np.array([x-y for x,y in zip(swapgrid[1:],swapgrid)])
            Pt_T=self.compute_discount_factor_from_rates(rates[idx],fixgrid[idx],swapgrid)
            swap_values=np.array([(P[0]-P[-1])/np.sum(delta*P[1:]) for P in Pt_T ])
            
            res_delta[i]=delta
            res_Pt_T[i]=Pt_T
            res_swap[i]=swap_values
            res_DF[i]=self.DF(fixgrid[idx])
            #undl[i]=(undl[i] - np.mean(undl[i],axis=0))/np.std(undl[i],axis=0)

        dic_arg={"swap":res_swap,
                 "Pt_T":res_Pt_T,
                 "delta":res_delta,
                 "DF":res_DF}

        if not include_rates:
            return dic_arg
        else:
            dic_arg.update({"rates":rates})
            return dic_arg