import numpy as np
from bisect import bisect
import QuantLib as ql
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

from Pricing.Curves import Instruments
from Pricing.Utilities import Data_File
from Pricing.Credit.Model import Intensity 
import Pricing.Credit.Instruments as Credit_instruments


def get_curves(calc_date:ql.Date,dic_df:dict,currency:str):
    mask=Data_File.select_row_from_keywords(dic_df['curve'],'Description',
                                                keywords=(currency,))
    items=Instruments.get_instruments(dic_df['curve'][mask],calc_date,
                                                  currency,'3M')
    curve=Curve(items,currency)
    issuer='CIC_'+currency
    entity=Credit_instruments.retrieve_credit_single_from_dataframe(dic_df['issuer'],issuer,calc_date)
    model_credit=Intensity.Calibration(curve,entity)
    risky_curve=Risky_Curve(curve,model_credit)
    return curve,risky_curve


def interval_integral(previous_r:float,r:float,times:tuple,t:float,eps=0.1) -> float:
    h=times[1]-times[0]
    delta=t-times[0]   
    if t<= times[0]+ eps*h:    
        res=delta*previous_r + (r-previous_r)*0.5*delta**2/(eps*h)
    else:
        res= eps*(h*previous_r + h*0.5*(r-previous_r)) + r*(1-eps)*(t-(times[0]+eps*h))
    
    return res
        
def get_grid(T:float,delta:float) -> np.ndarray:
    if T > delta:
        grid=np.arange(delta,T,delta)
    else:
        grid=np.array([delta,T])
        
    return grid

def find_first_zero(a) -> int:
    if 0 not in a:
        raise ValueError("No Zero")
    
    for i,x in enumerate(a):
        if x==0:
            return i

class Curve:
    """ Take Deposit_Instrument,Future_Instrument,Swap_Instrument to build Curve
    """
    def __init__(self,Instruments:list,cur_name:str):
        
        self.cur_name=cur_name
        #spot
        self.spot=Instruments[0].quote
        self.calibrate(Instruments)
    
    def from_Deposit(self,value:float,item:Instruments.Deposit)->float:
        """Retrieve rate from Deposit quote"""
        return (1-value)/(item.T*value)
    
    def from_Futures(self,value:float,delta:float) -> float:
        """Retrieve rate from Future quote"""
        return 100*(1+np.log(value)/delta)
        
    def from_Swap(self,r:float,item:Instruments.Swap) -> float:
        """Retrieve rate from Swap quote"""
        dic_freq={'1M':1/12,'3M':0.25,'6M':0.5,'1Y':1}
        
        last_idx=find_first_zero(self.rates[1:])
        t_last=self.timegrid[last_idx]
        
        h_float=dic_freq[item.float_delta]
        float_grid=get_grid(item.T,h_float)
        float_increment=interval_integral(self.rates[last_idx],r,(t_last,float_grid[-1]),float_grid[-1])
        float_DF=self.Discount_Factor(t_last)*np.exp(-float_increment)        
        
        h_fix=dic_freq[item.fix_delta]
        fix_grid=get_grid(item.T,h_fix)
        fix_prev_DF=np.array([ self.Discount_Factor(t) for t in fix_grid if t <= t_last])
        fix_increments=np.array([interval_integral(self.rates[last_idx],r,(t_last,t),t) for t in fix_grid if t> t_last])

        fix_DF= np.concatenate( (fix_prev_DF,self.Discount_Factor(t_last)*np.exp(-fix_increments)) )        
        LVL=h_fix*sum(fix_DF)
        
        return (1-float_DF)/LVL
    
    def parametric_form(self,t:float) ->float:
        
        idx=bisect(self.timegrid,t)
        if idx>len(self.timegrid)-1:
            idx=-1
        t1,t2=self.timegrid[idx-1],self.timegrid[idx]
        last_integral=interval_integral(self.rates[idx-1],self.rates[idx],(t1,t2),t)

        return self.value[idx-1]*np.exp(-last_integral)
    
    def Discount_Factor(self,t_arg):
                
        return self.interpolate(t_arg)
        
    def Forward(self,t_start:float,T:float,S:float)->float:
        """ Forward value of the rate between T and S starting at time t_start"""
        Pt_T=self.Discount_Factor(T)/self.Discount_Factor(t_start)
        Pt_S=self.Discount_Factor(S)/self.Discount_Factor(t_start)
    
        return (Pt_T/Pt_S-1)/(S-T) 

    def L(self,t:float,T:float) -> float:
        """ LIBOR """
        Pt_T=self.Discount_Factor(T)/self.Discount_Factor(t)
        delta=T-t
    
        return (1-Pt_T)/(delta*Pt_T)

    def Forward_Swap_rate(self,t:float,T:float,h_fix=1,h_float=0.5) -> float:
        """ Forward value of the swap of Term T starting at time t"""
        fix_grid=t+np.arange(0,T+h_fix,h_fix)
        P_fix=self.Discount_Factor(fix_grid[1:])/self.Discount_Factor(t)
        LVL=sum(P_fix)*h_fix

        float_grid=t+np.arange(0,T+h_float,h_float)
        P_float=self.Discount_Factor(float_grid)/self.Discount_Factor(t)

        return (P_float[0]-P_float[-1])/LVL
        
    def calibrate(self,items:list):
        
        self.timegrid=np.array([0]+[x.T for x in items])
        
        self.rates=np.zeros(len(self.timegrid))
        self.value=np.zeros(len(self.timegrid))
        self.value[0]=1
        for i,item in enumerate(items,1):
            times=(self.timegrid[i-1],self.timegrid[i])

            if (type(item)==Instruments.Deposit):
                ZC=lambda r : self.value[i-1]*np.exp(-interval_integral(self.rates[i-1],r,times,times[1]))
                func=lambda r: self.from_Deposit(ZC(r),item) - item.quote
                
            elif (type(item)==Instruments.Future):
                
                delta=times[1]-times[0]
                increment=lambda r: np.exp(-interval_integral(self.rates[i-1],r,times,times[1]))
                func=lambda r: self.from_Futures(increment(r),delta) - item.quote
                
            elif (type(item)==Instruments.Swap) :
                
                if ql.Period(item.period)<=ql.Period('1Y'):
                    ZC=lambda r : self.value[i-1]*np.exp(-interval_integral(self.rates[i-1],r,times,times[1]))
                    func=lambda r: self.from_Deposit(ZC(r),item) - item.quote
                else:
                    func=lambda r: self.from_Swap(r,item) -item.quote
                
            elif (item.category=='Basis Swap') :

                if ql.Period(item.period)<=ql.Period('1Y'):
                    ZC=lambda r : self.value[i-1]*np.exp(-interval_integral(self.rates[i-1],r,times,times[1]))
                    func=lambda r: self.from_Deposit(ZC(r),item) - item.quote
                else:
                    func=lambda r: self.from_Swap(r,item) -item.quote
                                
            self.rates[i]=brentq(func,-0.5,0.5,maxiter=100,xtol=1e-05)
            self.value[i]=self.value[i-1]*np.exp(-interval_integral(self.rates[i-1],self.rates[i],times,times[1]))

            self.interpolate=CubicSpline(self.timegrid,self.value)


class Risky_Curve:

    def __init__(self,curve:Curve,model:Intensity.Intensity):
        
        self.curve=curve
        self.model=model

    def Discount_Factor(self,tgrid:np.ndarray,risky:bool):
        R=self.model.entity.R
        DF=self.curve.Discount_Factor(tgrid)
        if not risky:
            return DF
        else:
             default_proba=np.array([1-self.model.compute_survival_proba(t) for t in tgrid ])
             return DF*(1-default_proba*(1-R))
        
    def adjustment(self,t:float,T: np.ndarray):
        R=self.model.entity.R
        default_proba=self.model.compute_survival_proba(t)-self.model.compute_survival_proba(T)

        return (1-default_proba*(1-R))