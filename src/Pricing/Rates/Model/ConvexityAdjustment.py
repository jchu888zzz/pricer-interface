import numpy as np
import QuantLib as ql
import re

from Pricing.Rates.Instruments import Swaption
import Pricing.Rates.Instruments as Rate_Instruments
from Pricing.Curves import Classic

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

#Works for ATM swaption
class Helper:

    def __init__(self,curve:Classic.Curve,instruments:list[Rate_Instruments.Swaption]):
        self.curve=curve
        self.calc_date=curve.calc_date
        self.get_values(instruments)

    def get_values(self,instruments:list[Rate_Instruments.Swaption]):
        self.dic_values=dict()
        day_counter = ql.Thirty360(ql.Thirty360.BondBasis)
        tenors=list(set([x.tenor for x in instruments]))
        for tenor in tenors:
            # Create tuples of (period, vol, expiry) for single-pass processing
            data = [(ql.Period(x.expiry), x.vol,x.typequote) for x in instruments if x.tenor==tenor]
            # Sort by period (avoids recreating Period objects during sort)
            data.sort(key=lambda x: x[0])
            # Extract sorted values
            self.dic_values[tenor]={'vol':np.array([x[1] for x in data], dtype=np.float64),
                                    'grid':np.array([day_counter.yearFraction(self.calc_date,self.calc_date + x[0])
                                                    for x in data], dtype=np.float64),
                                    'typequote':np.array([x[2] for x in data], dtype=str)}
        
    def compute_adjustment(self,t_array,tenor:str,delta_fix:float=1):

        if tenor not in self.dic_values.keys():
            raise ValueError(f'{tenor} not in current values')

        # Convert scalar to array for uniform processing
        is_scalar = np.isscalar(t_array)
        t_array = np.atleast_1d(t_array)

        vol=self.dic_values[tenor]['vol']
        grid=self.dic_values[tenor]['grid']
        typequote=self.dic_values[tenor]['typequote']

        idx=np.searchsorted(grid,t_array,side="left")
        tenor_period=convert_period(tenor)
        fix_tgrid=np.array([t + np.arange(0, tenor_period + delta_fix/2, delta_fix) for t in t_array])
        fix_zc=self.curve.discount_factor_from_times(fix_tgrid)
        delta=np.diff(fix_tgrid, axis=1)
        lvl=np.sum(delta*fix_zc[:,1:], axis=1)
        swap_value=(fix_zc[:,0]-fix_zc[:,-1])/lvl

        # Number of periods in the swap
        n_periods = fix_tgrid.shape[1] - 1

        # Vectorize the conditional logic
        vol_selected = vol[idx]
        typequote_selected = typequote[idx]

        # Create boolean masks for each type
        is_normal_vol = typequote_selected == 'normal_vol'
        is_vol = typequote_selected == 'vol'

        # Pre-compute common term
        common_term = func_Gprime(swap_value, n_periods) * lvl

        # Compute adjustment based on type
        adjustment = np.where(
            is_normal_vol,
            common_term * vol_selected**2 * t_array,
            np.where(
                is_vol,
                common_term * (swap_value)**2 * (np.exp(vol_selected**2 * t_array) - 1),
                np.nan  # Will raise error if we have invalid types
            )
        )

        # Check for invalid types
        if not np.all(is_normal_vol | is_vol):
            raise ValueError('Not implemented')

        # Return scalar if input was scalar
        return adjustment[0] if is_scalar else adjustment

