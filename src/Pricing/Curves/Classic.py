import pandas as pd
import QuantLib as ql
import numpy as np

from scipy.optimize import brentq
from Pricing.Credit.Model import Intensity 
import Pricing.Credit.Instruments as Credit_instruments
from Pricing.Utilities import Functions

def get_curves(calc_date:ql.Date,dic_df:dict,currency:str):
    instruments=sort_and_select_instruments(calc_date,dic_df['curve'],currency,'3M')
    curve=Curve(calc_date,currency,instruments)
    issuer='CIC_'+currency
    entity=Credit_instruments.retrieve_credit_single_from_dataframe(dic_df['issuer'],issuer,calc_date)
    model_credit=Intensity.Calibration(curve,entity)
    risky_curve=Risky_Curve(curve,model_credit)
    return curve,risky_curve


_DIC_DEPOSIT={'O_N':'1D','T_N':'2D','S_N':'3D','1M':'1M','2M':'2M','3M':'3M',
                '6M':'6M','9M':'9M','12M':'12M'}
class Deposit:
    def __init__(self,period:str,quote:float):
        self.period=period
        self.quote=quote
    
    def __repr__(self):
        return f'Deposit (Period:{self.period},quote:{self.quote})'
    
    def convert_period(self):
        return _DIC_DEPOSIT[self.period]


_DIC_MONTHS={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
class Future:
    def __init__(self,month:str,year:int,quote:float):
        self.month=month
        self.year=year
        self.quote=quote

    def __repr__(self):
        return f'Futures (Delivery:{self.month+str(self.year)},quote:{self.quote})'
    
    def get_start_date(self):
        """ Starts at third wednesday"""
        month=_DIC_MONTHS[self.month]
        res=ql.Date(15,month,self.year)
        if res.weekday()!=4:
            day= 15 + (4-res.weekday())%7
            return ql.Date(day,month,self.year)
        else:
            return res

_DIC_FREQ_SWAP={"EUR":{'Overnight':('1Y','1Y'),'1M':('1Y','1M'),'3M':('1Y','3M'),
                        '6M':('1Y','6M'),'Classical':('1Y','6M')},
                "USD":{'Overnight':('1Y','1Y'),'1M':('3M','3M'),'3M':('6M','3M'),
                        '6M':('6M','6M'), 'Classical':('1Y','6M')}}
class Swap:
    def __init__(self,period:str,quote:float,fix_freq:str,float_freq:str):
        
        self.period=period
        self.quote=quote
        self.fix_freq,self.float_freq=fix_freq,float_freq
    
    def __repr__(self):
        return f'Swap (Period:{self.period},quote:{self.quote})'

_PATTERNS={'Overnight': {'deposit':'Deposit', 'swap':r'^(?=.*Basis_swap)(?=.*V_Overnight)'},
    '3M': {'deposit':r'^(?=.*Deposit)(?=.*3M)', 'future':r'^(?=.*futures)(?=.*3M)', 'swap':r'^(?=.*Basis_swap)(?=.*V_3M)'},
    'Classical': {'deposit':'Deposit', 'future':r'^(?=.*futures)(?=.*3M)', 'swap':'Swap'},
    '6M': {'deposit':r'^(?=.*Deposit)(?=.*6M)', 'swap':r'^(?=.*Basis_swap)(?=.*V_6M)'},
    '1M': {'deposit':r'^(?=.*Deposit)(?=.*1M)', 'swap':r'^(?=.*Basis_swap)(?=.*V_1M)'}}


_BUSINESS_CALENDAR=ql.TARGET()
_CRITERION={'EUR': (ql.Period("8M"),ql.Period("18M")),
                    'USD':(ql.Period("5M"),ql.Period("12M")),
                    'GBP':(ql.Period("9M"),ql.Period("20M")),
                    'CHF':(ql.Period("9M"),ql.Period("18M")) }

def sort_and_select_instruments(calc_date:ql.Date,df:pd.DataFrame,cur_name:str,option='3M') -> list  :
    """Retrieve Curve Instruments from a formatted dataframe
        Returns: list of Instruments  Deposits/Future/Swaps """

    def make_deposit(item):
        name=item[0].split()
        res=Deposit(period=name[-1],quote=item[1])
        period=res.convert_period()
        res.maturity_date=_BUSINESS_CALENDAR.advance(calc_date,ql.Period(period),ql.ModifiedFollowing)
        return res

    fix_freq,float_freq=_DIC_FREQ_SWAP[cur_name][option]    
    def make_swap(item):
        if '@' in item[0]:
            name=item[0].split('@')[0].split()
        else:
            name=item[0].split()
        res=Swap(period=name[-1],quote=item[1],fix_freq=fix_freq,float_freq=float_freq)
        res.maturity_date=_BUSINESS_CALENDAR.advance(calc_date+ql.Period("2D"),ql.Period(res.period),
                                            ql.ModifiedFollowing)
        res.fix_schedule=list(ql.MakeSchedule(calc_date+ql.Period("2D"),
                                            res.maturity_date,ql.Period(fix_freq)))
        res.float_schedule=list(ql.MakeSchedule(calc_date+ql.Period("2D"),
                                                res.maturity_date,ql.Period(float_freq)))
        return res

    def make_future(item):
        name=item[0].split()
        if '@' in item[0]:
            month,year=name[-3],int(name[-2])
        else:
            month,year=name[-2],int(name[-1])
        res=Future(month=month,year=year,quote=item[1])
        res.start_date=res.get_start_date()
        res.maturity_date=_BUSINESS_CALENDAR.advance(res.start_date,ql.Period("3M"),ql.ModifiedFollowing)
        return res
    
    patterns=_PATTERNS.get(option,{})
    res=[]
    df=df[df.Description.str.contains(cur_name)]
    if 'deposit' in patterns:
        mask_deposit=df.Description.str.contains(patterns['deposit'])
        deposit_items=list(map(make_deposit,df[mask_deposit].to_numpy()))
        deposit_items=[x for x in deposit_items if x.maturity_date <= calc_date+_CRITERION[cur_name][0]]
        if deposit_items:
            res.extend(deposit_items)

    if 'future' in patterns:
        mask_future=df.Description.str.contains(patterns['future'])
        future_items=list(map(make_future,df[mask_future].to_numpy()))
        future_items=[x for x in future_items if (calc_date+_CRITERION[cur_name][0] < x.maturity_date 
                                                    < calc_date+_CRITERION[cur_name][1])]
        if future_items:
            res.extend(future_items)
    if 'swap' in patterns:
        mask_swap=df.Description.str.contains(patterns['swap'])
        swap_items=list(map(make_swap,df[mask_swap].to_numpy()))
        swap_items=[x for x in swap_items if calc_date+_CRITERION[cur_name][1] < x.maturity_date ]
        if swap_items:
            res.extend(swap_items)

    return sorted(res,key=lambda x: x.maturity_date)

def find_first_zero(a) -> int:
    if 0 not in a:
        raise ValueError("No Zero")
    for i,x in enumerate(a):
        if x==0:
            return i

class Curve:
    """ Take Deposit,Future,Swap to build Curve
    """
    def __init__(self,calc_date:ql.Date,cur_name:str,instruments:list):
        
        self.cur_name=cur_name
        self.calc_date=calc_date
        #spot
        #self.spot=Instruments[0].quote
        self.calendar=ql.Actual360()
        self.calibrate(instruments)
    
    def convert_dates_to_tgrid(self,dates:list[ql.Date]):
        return [self.calendar.yearFraction(self.calc_date,d) for d in dates]
    
    def parametric_form(self,t_array:np.ndarray) ->float:
        return np.exp(-self.precomputed_integral.evaluate(t_array))
    
    def calibrate(self,instruments:list):
        eps=0.1
        def zc_temp(r:float,t:float):
            temp_rates=self.rates.copy()
            idx=Functions.first_occ(temp_rates,0)
            temp_rates[idx:]=r
            return np.exp(-Functions.integral_cst_by_part(temp_rates, self.tgrid,t, eps))
        
        def from_deposit(r:float,T:float)->float:
            value=zc_temp(r,T)
            return (1-value)/(T*value)
        
        def from_futures(value:float,t:float,T:float):
            return 100*(1+np.log(value)/(T-t))
        
        def from_swap(r:float,fix_schedule:list[ql.Date],float_schedule:list[ql.Date]):
            fix_grid=np.array([self.calendar.yearFraction(self.calc_date,x)
                                for x in fix_schedule])
            
            fix_DF=zc_temp(r, fix_grid)      
            lvl=sum(fix_DF[1:]*np.array([x-y for x,y in zip(fix_grid[1:],fix_grid)]))

            t_float=self.calendar.yearFraction(self.calc_date,float_schedule[-1]) 
            float_DF=zc_temp(r, t_float)      
            return (1-float_DF)/lvl

        self.tgrid=np.array([self.calendar.yearFraction(self.calc_date,x.maturity_date)
                                    for x in instruments])        
        self.rates=np.zeros(len(self.tgrid))
        self.value=np.zeros(len(self.tgrid))
        for i,item in enumerate(instruments):
            t=self.tgrid[i]
            if isinstance(item,Deposit):
                func=lambda r: from_deposit(r,t) - item.quote
                
            elif isinstance(item,Future):
                t0=self.calendar.yearFraction(self.calc_date,item.start_date)
                t1=self.calendar.yearFraction(self.calc_date,item.maturity_date)
        
                func=lambda r: from_futures(zc_temp(r,t)/self.value[i-1],t0,t1) - item.quote
                
            elif isinstance(item,Swap) :
                if ql.Period(item.period)<ql.Period('1Y'):
                    func=lambda r: from_deposit(r,t) - item.quote
                else:
                    func=lambda r: from_swap(r,item.fix_schedule,
                                            item.float_schedule) -item.quote
            else:
                raise ValueError(f'{item} instance not valid')
            
            self.rates[i]=brentq(func,-0.5,0.5,maxiter=100,xtol=1e-05)
            self.value[i]=np.exp(-Functions.integral_cst_by_part(self.rates, self.tgrid,t,eps))

        self.precomputed_integral=Functions.IntegralCSTPrecalculated(self.rates,self.tgrid,eps)

    def discount_factor(self,dates:list[ql.Date]):
        tgrid=self.convert_dates_to_tgrid(dates)
        return self.parametric_form(tgrid)
    
    def discount_factor_from_times(self,t_array:np.ndarray):
        return self.parametric_form(t_array)

    def forward_swap_rate(self,dates:list[ql.Date] | ql.Date,tenor:str,
                            fix_freq:str,float_freq:str) -> float:
        """ Forward value of the swap starting at time t"""
        
        is_single_point= isinstance(dates,ql.Date)

        if is_single_point:
            dates=[dates]
        
        res=np.zeros(len(dates))
        for i,d in enumerate(dates):
            start=d+ql.Period('2D')
            fix_schedule= list(ql.MakeSchedule(start,start+ql.Period(tenor),ql.Period(fix_freq)))
            fix_grid=np.array([self.calendar.yearFraction(self.calc_date,x)
                                for x in fix_schedule])
            P_fix=self.discount_factor(fix_schedule)
            lvl=sum(P_fix[1:]*np.array([x-y for x,y in zip(fix_grid[1:],fix_grid)]))
            
            float_schedule= list(ql.MakeSchedule(start,start+ql.Period(tenor),ql.Period(float_freq)))
            P_float=self.discount_factor(float_schedule)
            res[i]=(P_float[0]-P_float[-1])/lvl
        if is_single_point:
            return res[0]
        return res
    

class Risky_Curve:

    def __init__(self,curve:Curve,model:Intensity.Intensity):
        self.curve=curve
        self.model=model
        self.calendar=curve.calendar
        self.calc_date=curve.calc_date
        self.get_funding_spread_interp()
        
    def discount_factor(self,dates:list[ql.Date],risky:bool):
        Recovery=self.model.entity.R
        zc=self.curve.discount_factor(dates)
        if not risky:
            return zc
        else:
            tgrid=[self.calendar.yearFraction(self.calc_date,d) for d in dates]
            default_proba=np.array([1-self.model.compute_survival_proba(t) for t in tgrid ])
            return zc*(1-default_proba*(1-Recovery))
        
    def adjustment(self,t:float,T: np.ndarray):
        Recovery=self.model.entity.R
        default_proba=self.model.compute_survival_proba(t)-self.model.compute_survival_proba(T)
        return (1-default_proba*(1-Recovery))
    
    def get_funding_spread_interp(self):

        entity=self.model.entity
        grid=[self.calendar.yearFraction(self.calc_date,self.calc_date+ql.Period(p)) for p in entity.tenors]
        self.funding_spread_interp= lambda x : np.interp(x,grid,entity.quotes,left=entity.quotes[0],right=entity.quotes[-1])