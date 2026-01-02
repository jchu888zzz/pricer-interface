import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq,least_squares
import QuantLib as ql
from scipy.interpolate import CubicSpline

from Pricing.Utilities import Dates
from Pricing.Utilities.decorators import timer 
import Pricing.Rates.Model.ConvexityAdjustment as CA
from Pricing.Utilities.InputConverter import convert_period
import Pricing.Rates.Instruments as Rate_Instruments
import Pricing.Utilities.Functions as Functions
from Pricing.Curves import Classic


_DIC_FREQ_SWAPTION={"EUR":{"delta_fix":1,"delta_float":0.5},
                    "USD":{"delta_fix":0.5,"delta_float":0.25}}

def get_model(calc_date:ql.Date,mkt_data:dict,currency:str,option='swaption') -> dict:
    calc_date=ql.Date.todaysDate()
    curve,risky_curve=Classic.get_curves(calc_date,mkt_data,currency)

    if option=='swaption':
        instruments=Rate_Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                            curve,calc_date,currency)
        instruments=[x for x in instruments if x.strike_type=='ATM']
    elif option=="cap":
        instruments=Rate_Instruments.select_and_prepare_caps(mkt_data['caps'],curve,calc_date,currency)
    else:
        raise ValueError(f"{option} not implemented")

    model=Calibration(curve,instruments)
    return {'risky_curve':risky_curve,
            'curve':curve,
            'model':model,
            'calc_date':calc_date}

def Put(vol:float,K:float,ZC1:float,ZC2:float)-> float:
    
    d1=(1/vol)*np.log(ZC1*K/ZC2)+ 0.5*vol
    d2=d1-vol
    return K*ZC1*norm.cdf(d1) - ZC2*norm.cdf(d2)

@timer
def Calibration(curve,instru_calib:list[Rate_Instruments.Swaption | Rate_Instruments.Cap]):
    Caps=[x for x in instru_calib if isinstance(x,Rate_Instruments.Cap)]
    Swaptions=[x for x in instru_calib if isinstance(x,Rate_Instruments.Swaption) ]
    def error_function(param:tuple[float]):
        model=HW(curve,param=param)
        res=[]
        for item in Swaptions:
            res.append((item.mkt_price-model.price_swaption(item))/1e4)
        for item in Caps:
            res.append((item.mkt_price-model.price_cap(item))/1e4)

        return np.array(res)

    optimizer=least_squares(error_function,x0=(0.02,0.02),bounds=([1e-5,1e-3],[0.5,np.sqrt(0.1)]))

    return HW(curve,optimizer.x)

def select_rates(rates:np.ndarray,simu_dates:np.ndarray[ql.Date],fix_dates:np.ndarray[ql.Date],
                nb_sub_fix_points:None| int) -> np.ndarray:
    if not nb_sub_fix_points:
        idx=Functions.find_idx(simu_dates,fix_dates)
        return rates[:,idx].T
    else:
        res=[]
        for i,(d1,d2) in enumerate(zip(fix_dates,fix_dates[1:])):
            sub_schedule=Dates.ql_linspace(d1,d2,nb_sub_fix_points)
            sub_idx=Functions.find_idx(simu_dates,sub_schedule)
            res.append(rates[:,sub_idx].T)
        return np.array(res)

class HW :
    
    def __init__(self,curve:Classic.Curve,param=(0.02,0.02)):
        #Forward and DF are interpolation function
        self.curve=curve
        #self.DF=curve.discount_factor_from_times
        self.DF=CubicSpline(curve.tgrid,curve.value)
        self.a,self.sigma=param[0],param[1]
        
    def __repr__(self):
        return f'HW(a:{self.a},sigma:{self.sigma})'
    
    #Smoother instantaneous forward rate
    def instantaneous_f(self,t,h=0.1):
        res=-(np.log(self.DF(t+h))-np.log(self.DF(t)))/h
        return res
    
    def compute_B_term(self,delta:np.ndarray|float,factor=1) -> np.ndarray|float:
        if (self.a==0):
            return delta
        else:
            a=factor*self.a
            return (1-np.exp(-a*delta))/a

    def affine_term(self,t:float,T:np.ndarray|float) ->tuple[np.ndarray]:
        forward=self.instantaneous_f(t)
        B_term=self.compute_B_term(T-t)
        A_term= self.DF(T)/self.DF(t)*np.exp(B_term*forward - 0.5*(B_term**2)*self.sigma**2*self.compute_B_term(t,factor=2))
        return (A_term,B_term)
    
    def compute_discount_factor_from_rates(self,rate:np.ndarray,t:float,T:np.ndarray|float)->np.ndarray:
        A_term,B_term=self.affine_term(t,T)
        rate_reshaped=rate.reshape(-1,1) if rate.ndim==1 else rate
        return A_term*np.exp(-rate_reshaped*B_term)
    
    def var_(self,t:np.ndarray|float) -> np.ndarray|float:
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
    
    def price_cap(self,item:Rate_Instruments.Cap)->float:
        
        K_prime=(1+item.K*item.delta)
        ZC_ratio=[x/y for x,y in zip(item.ZC,item.ZC[1:])]
        
        vol_i=self.compute_B_term(item.delta)*self.sigma*np.sqrt(self.compute_B_term(item.tgrid[:-1],factor=2))
        d1=(1/vol_i)*np.log(ZC_ratio/K_prime) + 0.5*vol_i
        d2=d1-vol_i
    
        return np.sum(item.ZC[:-1]*norm.cdf(d1)-K_prime*item.ZC[1:]*norm.cdf(d2))*10**4
    
    def price_swaption(self,item:Rate_Instruments.Swaption)->float:
        c=item.delta.copy()*item.K
        c[-1]+=1
        
        A_term,B_term=self.affine_term(item.tgrid[0],item.tgrid[1:])
        func_to_solve= lambda x : np.sum(c*A_term*np.exp(-B_term*x)) -1 
        r_star=brentq(func_to_solve,-0.8,0.8)
        X_term=A_term*np.exp(-B_term*r_star)

        vol_=B_term*self.sigma*np.sqrt(self.compute_B_term(item.tgrid[0],factor=2))
        return np.sum(c*Put(vol_,X_term,item.ZC[0],item.ZC[1:]))*10**4

    def get_CMS_adjustment(self,Curve,calc_date:ql.Date,Instruments:list):
        self.cms_adj=CA.GetAdjustment(Curve,calc_date,Instruments)
    
    #52 = weeks,360 days
    def rate_simulation(self,T:float,grid:list,Nb_simu:int,rng:np.random._generator.Generator):
        
        alpha,Var=[self.alpha_T(t,T) for t in grid ],[self.var_(t) for t in grid]
        res=np.zeros((Nb_simu,len(grid)))
        initial_t=0.25
        res[:,0]=-(np.log(self.DF(initial_t))-np.log(self.DF(0)))/initial_t

        for i in range(1,len(grid)):
            h=grid[i]-grid[i-1]
            res[:,i]=( res[:,i-1]*np.exp(-self.a*h) + alpha[i] - alpha[i-1]*np.exp(-self.a*h) 
                        + np.sqrt(Var[i]-Var[i-1]*np.exp(-2*self.a*h))*rng.standard_normal(size=Nb_simu) )
        
        return res
    
    def generate_rates(self,calc_date:ql.Date,maturity_date:ql.Date,
                        cal=ql.Business252(),
                        Nbsimu=10000,seed=42) -> dict:
        rng=np.random.default_rng(int(seed))
        T_maturity=cal.yearFraction(calc_date,maturity_date)
        schedule=Dates.compute_target_schedule(calc_date,maturity_date,ql.Period('1D'))
        grid=np.array([cal.yearFraction(calc_date,x) for x in schedule[1:] ])
        rates=self.rate_simulation(T_maturity,grid,Nbsimu,rng)
        return {'rates':rates,'schedule':schedule}
    
    def compute_deposit_from_rates(self,rates:np.ndarray,t:float,tenor:str) -> np.ndarray:
        h=convert_period(tenor)
        P_term=self.compute_discount_factor_from_rates(rates,t,t+h)
        return (1-P_term)/(P_term*h)
    
    def compute_cms_from_rates(self,rates:np.ndarray,t:float,tenor:str,delta_fix:float,delta_float:float) -> np.ndarray:
        tenor=convert_period(tenor)
        fix_tgrid=t+np.arange(0,tenor,delta_fix)
        P_fix=self.compute_discount_factor_from_rates(rates,t,fix_tgrid)
        delta=np.array([x-y for x,y in zip(fix_tgrid[1:],fix_tgrid)])
        lvl=np.sum(P_fix[:,1:]*delta,axis=1)
        
        float_tgrid=t+np.arange(0,tenor,delta_float)
        P_float=self.compute_discount_factor_from_rates(rates,t,float_tgrid)
        res=(P_float[:,0]-P_float[:,-1])/lvl
        return res
    
    #wrapper to select rates
    def select_rates(self,data_rates:dict,fix_dates:list[ql.Date]) -> np.ndarray:
        return select_rates(data_rates['rates'],data_rates['schedule'],fix_dates)
    
    def compute_single_undl_from_rates(self,data_rates:dict,fix_dates:list[ql.Date],undl1:str,include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""
        cur1,rate_type1,tenor1=undl1.split()
        rates=select_rates(data_rates['rates'],data_rates['schedule'],fix_dates,None)
        fixgrid=[self.curve.calendar.yearFraction(self.curve.calc_date,d) for d in fix_dates]
        if rate_type1=="CMS":
            undl=np.array([self.compute_cms_from_rates(rates[i],t,tenor1,
                                                       _DIC_FREQ_SWAPTION[cur1]["delta_fix"],
                                                       _DIC_FREQ_SWAPTION[cur1]["delta_float"])
                                                         for i,t in enumerate(fixgrid)])
        elif rate_type1=="Euribor":
            undl=np.array([self.compute_deposit_from_rates(rates[i],t,tenor1) for i,t in enumerate(fixgrid)])
        else:
            raise ValueError(f"{rate_type1} not implemented")
        if not include_rates:
            return {'undl':undl,'nbsimu':rates.shape[1]}
        else:
            return {'undl':undl,'nbsimu':rates.shape[1],'rates':rates}
    
    def compute_single_undl_from_rates_with_depth(self,data_rates:dict,fix_dates:list[ql.Date],undl1:str,nb_sub_fix_points:int,
                                                    include_rates=True)->np.ndarray:
        """ result shape (len(fixgrid),fixing_depth,nb simu)"""
        cur1,rate_type1,tenor1=undl1.split()
        rates=select_rates(data_rates['rates'],data_rates['schedule'],fix_dates,nb_sub_fix_points)
        res=[]
        
        for i,(d1,d2) in enumerate(zip(fix_dates,fix_dates[1:])):
            sub_schedule=Dates.ql_linspace(d1,d2,nb_sub_fix_points)
            sub_fixgrid=(self.curve.calendar.yearFraction(self.curve.calc_date,d) for d in sub_schedule)
            if rate_type1=="CMS":
                res.append(np.array([ self.compute_cms_from_rates(rates[i][j],t,tenor1,
                                                                _DIC_FREQ_SWAPTION[cur1]["delta_fix"],
                                                                _DIC_FREQ_SWAPTION[cur1]["delta_float"]) 
                                for j,t in enumerate(sub_fixgrid)]))
            elif rate_type1=="Euribor":
                res.append(np.array([ self.compute_deposit_from_rates(rates[i][j],t,tenor1) 
                                for  j,t in enumerate(sub_fixgrid)]))
        
        if not include_rates:
            return {'undl':np.array(res),'nbsimu':data_rates['rates'].shape[1]}
        else:
            return {'undl':np.array(res),'nbsimu':data_rates['rates'].shape[1],'rates':rates[:,-1,:]}
            
    def compute_prep_for_swaption_from_rates(self,contract,data_rates:dict,
                                            daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis),
                                            include_rates=True) ->dict:
        """ result shape (len(fixgrid),nb simu)"""

        fix_dates=contract.fix_dates
        call_dates=contract.call_dates
        maturity_date=contract.pay_dates[-1]
        currency=contract.currency
        fix_freq=_DIC_FREQ_SWAPTION[currency]["delta_fix"]
        float_freq=_DIC_FREQ_SWAPTION[currency]["delta_float"]


        rates=select_rates(data_rates['rates'],data_rates['schedule'],fix_dates,None)
        res_delta=[None]*len(call_dates)
        res_Pt_T=[None]*len(call_dates)
        res_swap=[None]*len(call_dates)
        res_DF=[None]*len(call_dates)

        calc_date=self.curve.calc_date

        for i,d in enumerate(call_dates):
            idx=Functions.find_idx(fix_dates,d)
            t=daycount_calendar(calc_date,d)
            fix_schedule= list(ql.MakeSchedule(fix_dates[idx],maturity_date,ql.Period(fix_freq)))
            fix_tgrid=np.array([daycount_calendar.yearFraction(calc_date,x) for x in fix_schedule])
            P_fix=self.compute_discount_factor_from_rates(rates[idx],t,fix_tgrid)
            delta=np.array([x-y for x,y in zip(fix_tgrid[1:],fix_tgrid)])
            lvl=np.sum(P_fix[1:]*delta,axis=1)
            
            float_schedule= list(ql.MakeSchedule(fix_dates[idx],maturity_date,ql.Period(float_freq)))
            float_tgrid=np.array([daycount_calendar.yearFraction(calc_date,x) for x in float_schedule])
            P_float=self.compute_discount_factor_from_rates(rates[idx],t,float_tgrid)
            swap_values=(P_float[:,0]-P_float[:,-1])/lvl
            
            res_delta[i]=delta
            res_Pt_T[i]=P_float
            res_swap[i]=swap_values
            res_DF[i]=self.DF(t)

        dic_arg={"swap":res_swap,
                "Pt_T":res_Pt_T,
                "delta":res_delta,
                "DF":res_DF,
                'nbsimu':rates.shape[1]}

        if not include_rates:
            return dic_arg
        else:
            dic_arg.update({"rates":rates})
            return dic_arg
