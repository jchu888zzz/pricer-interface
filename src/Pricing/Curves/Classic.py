import pandas as pd
import QuantLib as ql
import numpy as np

from scipy.optimize import brentq
from Pricing.Credit.Model import Intensity 
import Pricing.Credit.Instruments as Credit_instruments


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
_CRITERION={'EUR': (ql.Period("9M"),ql.Period("23M")),
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
                                            res.maturity_date,ql.Period(fix_freq)))[1:]
        res.float_schedule=list(ql.MakeSchedule(calc_date+ql.Period("2D"),
                                                res.maturity_date,ql.Period(float_freq)))[1:]
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

def interval_integral(previous_r, r, t1, t_array, t2, eps=0.1):
    """
    Compute interval integral - handles both scalar and array inputs.

    Args:
        previous_r: Previous rate (scalar or array)
        r: Current rate (scalar or array)
        t1: Start time (scalar or array)
        t_array: Time points to evaluate (scalar or array)
        t2: End time (scalar or array)
        eps: Parameter (default 0.1)

    Returns:
        Scalar or array of integral values matching input shape
    """
    # Handle scalar inputs
    is_scalar = np.isscalar(t_array)
    if is_scalar:
        t_array = np.array([t_array])
        previous_r = np.atleast_1d(previous_r)
        r = np.atleast_1d(r)
        t1 = np.atleast_1d(t1)
        t2 = np.atleast_1d(t2)

    h = t2 - t1
    t_delta = t_array - t1
    r_delta = r - previous_r

    # Vectorized conditional: use np.where to branch based on condition
    threshold = t1 + eps * h
    # Calculate both branches
    early_branch = t_delta * previous_r + r_delta * 0.5 * t_delta ** 2 / (eps * h)
    late_branch = eps * (h * previous_r + h * 0.5 * r_delta) + r * (1 - eps) * (t_array - threshold)
    # Select based on condition
    result = np.where(t_array <= threshold, early_branch, late_branch)

    # Return scalar if input was scalar
    if is_scalar:
        return result[0]
    return result

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

        # Handle scalar input - check both np.isscalar and single-element arrays
        is_scalar = np.isscalar(t_array) or (isinstance(t_array, np.ndarray) and t_array.size == 1)
        if is_scalar:
            # Return 1.0 for negative time values (no discounting)
            if t_array < 0:
                return 1.0
            t_array = np.array([t_array])
        else:
            # For array inputs, handle negative values
            t_array = np.asarray(t_array)

        idx = np.searchsorted(self.tgrid, t_array)
        # Handle boundary condition: if idx > len(tgrid) - 1, set to -1
        idx = np.where(idx > len(self.tgrid) - 1, -1, idx)

        # Get t1, t2 for each evaluation point
        t1 = self.tgrid[idx - 1]
        t2 = self.tgrid[idx]

        # Get rates and values for previous indices
        r_prev = self.rates[idx - 1]
        r_curr = self.rates[idx]
        val_prev = self.value[idx - 1]

        # Compute last integral for all t values at once
        # Note: Here we call our vectorized interval_integral
        last_integral = interval_integral(r_prev, r_curr, t1, t_array, t2)

        result = val_prev * np.exp(-last_integral)

        # Handle negative values for array inputs: return 1.0 for t < 0
        result = np.where(t_array <= 0, 1.0, result)

        # Return scalar if input was scalar
        if is_scalar:
            return result[0]
        return result
    
    def calibrate(self,instruments:list):
        
        def from_deposit(value:float,T:float)->float:
            return (1-value)/(T*value)
        
        def from_futures(value:float,t:float,T:float):
            return 100*(1+np.log(value)/(T-t))
        
        def from_swap(r_prev:float,r:float,t_prev:float,
                    fix_schedule:list[ql.Date],float_schedule:list[ql.Date]):
            fix_grid=np.array([self.calendar.yearFraction(self.calc_date,x)
                                for x in fix_schedule])
            fix_prev_DF=np.array([ self.parametric_form(t) for t in fix_grid if t <= t_prev])
            fix_increments=np.array([interval_integral(r_prev,r,t_prev,t,t) 
                                        for t in fix_grid if t> t_prev])
            fix_DF= np.concatenate( (fix_prev_DF,self.parametric_form(t_prev)*np.exp(-fix_increments)) )        

            lvl=sum(fix_DF*np.array([fix_grid[0]]+ [x-y for x,y in zip(fix_grid[1:],fix_grid)]))
            
            float_grid=np.array([self.calendar.yearFraction(self.calc_date,x)
                                for x in float_schedule])

            float_increment=interval_integral(r_prev,r,t_prev,float_grid[-1],float_grid[-1])
            float_DF=self.parametric_form(t_prev)*np.exp(-float_increment)     
            return (1-float_DF)/lvl
        
        self.tgrid=np.array([0]+[self.calendar.yearFraction(self.calc_date,x.maturity_date)
                                    for x in instruments])        
        self.rates=np.zeros(len(self.tgrid))
        self.value=np.zeros(len(self.tgrid))
        self.value[0]=1
        for i,item in enumerate(instruments,start=1):
            t_prev=self.tgrid[i-1]
            t=self.tgrid[i]
            r_prev=self.rates[i-1]

            if isinstance(item,Deposit):
                zc=lambda r : self.value[i-1]*np.exp(-interval_integral(r_prev,r,t_prev,t,t))
                func=lambda r: from_deposit(zc(r),t) - item.quote
                
            elif isinstance(item,Future):
                t0=self.calendar.yearFraction(self.calc_date,item.start_date)
                t1=self.calendar.yearFraction(self.calc_date,item.maturity_date)
                increment=lambda r: np.exp(-interval_integral(r_prev,r,t0,t,t1))
                func=lambda r: from_futures(increment(r),t0,t1) - item.quote
                
            elif isinstance(item,Swap) :
                if ql.Period(item.period)<ql.Period('1Y'):
                    zc=lambda r : self.value[i-1]*np.exp(-interval_integral(r_prev,r,t_prev,t,t))
                    func=lambda r: from_deposit(zc(r),item) - item.quote
                else:
                    
                    func=lambda r: from_swap(r_prev,r,t_prev,item.fix_schedule,
                                            item.float_schedule) -item.quote
            else:
                raise ValueError(f'{item} instance not valid')
                
                                
            r=brentq(func,-0.5,0.5,maxiter=100,xtol=1e-05)
            self.value[i]=self.value[i-1]*np.exp(-interval_integral(r_prev,r,t_prev,t,t))
            self.rates[i]=r
            
            
    def discount_factor(self,dates:list[ql.Date]):
        tgrid=self.convert_dates_to_tgrid(dates)
        return self.parametric_form(tgrid)
    
    
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