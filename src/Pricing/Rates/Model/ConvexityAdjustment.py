import numpy as np
from scipy.interpolate import CubicSpline
import QuantLib as ql
import re

from Pricing.Rates.Instruments import Swaption

def convert_period(period:str) -> float:
    dic={'D':1/360,'W':7/360,'M':30/360,'Y':1}
    temp=re.split(r'(\D+)',period)
    return float(temp[0])*dic[temp[1]]

def func_G(x:float,n:int,q=1,delta=0.5) -> float:
    return  x/(1+x/q)**delta*1/(1-1/(1+x/q)**n)

def func_Gprime(x:float,n:int,q=1,delta=0.5)-> float:
    
    g1=x/(1+x/q)**delta    
    g1_prime=(1- x/q*delta/(1+x/q) )/(1+x/q)**delta
    
    g2=1/(1-1/(1+x/q)**n)
    g2_prime=-(n/q)*(1+x/q)**(-(n+1))/(1-(1+x/q)**(-n))**2
    
    return g1*g2_prime + g2*g1_prime

def cms_adjustment(curve,date:ql.Date,tenor:str,freq:str,vol_grid:np.ndarray,dates_grid:np.ndarray,type_vol='normal'):

    daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis)
    idx=np.searchsorted(dates_grid,date,side="left")
    vol=vol_grid[idx]
   
    fix_schedule=list(ql.MakeSchedule(date,date+ql.Period(tenor),ql.Period(freq)))
    fix_zc=curve.discount_factor(fix_schedule[1:])
    fix_tgrid=[daycount_calendar.yearFraction(date,d) for d in fix_schedule]
    delta=np.array([x-y for x,y in zip(fix_tgrid[1:],fix_tgrid)])
    lvl=sum(delta*fix_zc)

    swap_value=(fix_zc[0]-fix_zc[-1])/lvl

    


