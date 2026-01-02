import QuantLib as ql
import numpy as np
from scipy.stats import norm
import pandas as pd

from Pricing.Utilities.Display import truncate
from Pricing.Curves.Classic import Curve

class CMSSpreadOption:
    
    def __init__(self,Expiry,Term1,Term2,K,price):
        
        self.Expiry,self.T1,self.T2=Expiry,Term1,Term2
        self.K=K
        self.price=price
    
    def __repr__(self):
        
        return f'Spread_Instruments(Expiry:{self.Expiry},T1:{self.T1},T2:{self.T2},Strike:{truncate(self.K,3)},price:{truncate(self.price,2)})'

class Swaption:

    def __init__(self,parameters:tuple):
        self.vol,self.expiry,self.tenor=parameters[0],parameters[1],parameters[2] #Vol quote en  %
        self.typequote=parameters[3]
        self.float_freq=parameters[4]
        self.fix_freq=parameters[5]
        self.strike_type=parameters[6]
        self.strike_shift=parameters[7]
        
    def __repr__(self):
        if self.strike_type=='from_ATM':
            striketype=' '.join([self.strike_type,str(self.strike_shift)])
        else:
            striketype=self.strike_type
        return f'Swaption (Expiry:{self.expiry},Tenor:{self.tenor},striketype:{striketype},quote:{self.vol})'
    
    def compute_mkt_price(self,calc_date:ql.Date,curve:Curve,daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis)):
        
        start_date=calc_date + ql.Period(self.expiry)
        end_date=start_date + ql.Period(self.tenor)
        schedule=list(ql.MakeSchedule(start_date,end_date,ql.Period(self.float_freq)))
        tgrid=np.array([daycount_calendar.yearFraction(calc_date,x) for x in schedule])
        delta=np.array([x-y for x,y in zip(tgrid[1:],tgrid)])
        zc=curve.discount_factor(schedule)
        #Compute strike
        strike=curve.forward_swap_rate(start_date,self.tenor,self.fix_freq,self.float_freq)
        
        if self.strike_type=='from_ATM':
            strike+=float(self.strike_shift)
        
        fwd=curve.forward_swap_rate(start_date,self.tenor,self.fix_freq,self.float_freq)
        
        t=curve.calendar.yearFraction(calc_date,start_date)
        vol_=self.vol*np.sqrt(t)

        #Retrieve param to compute theorical price
        self.ZC=zc
        self.delta=delta
        self.K=strike
        self.tgrid=tgrid

        if (self.typequote== 'normal_vol'):
            k=(fwd-strike)/vol_    
            self.mkt_price=np.sum(zc[1:]*delta)*vol_*(np.exp(-0.5*k**2)/np.sqrt(2*np.pi)  + k*norm.cdf(k))*10000
        else:
            d1,d2=0.5*vol_,-0.5*vol_            
            self.mkt_price=np.sum(zc[1:]*delta)*strike*(norm.cdf(d1)-norm.cdf(d2))*10000

_DIC_FREQ_SWAPTION={'USD':{'fix_freq':'6M','float_freq':'3M'},
                'EUR':{'fix_freq':'1Y','float_freq':'6M'}}
def select_and_prepare_swaptions(df:pd.DataFrame,curve,calc_date:ql.Date,currency:str) -> tuple[Swaption]:
    
    mask=df.Description.str.contains(currency and 'Swaption')
    
    def make_swaption(item):
        name,quote=item[0].split(),item[1]
        t_indicator=[x for x in name if any(y.isdigit()==True for y in x)]
        expiry,maturity=t_indicator[0],t_indicator[1]

        if 'from_ATM' in name:
            striketype='from_ATM'
            strike_shift=name[-1]
        else:
            striketype='ATM'
            strike_shift=0

        if 'normal' in name[0]:
            type_quote='normal_vol'
        else :
            type_quote='vol'
        param=(quote,expiry,maturity,type_quote,_DIC_FREQ_SWAPTION[currency]['float_freq'],
                _DIC_FREQ_SWAPTION[currency]['fix_freq'],striketype,strike_shift)
        res=Swaption(param)
        res.compute_mkt_price(calc_date,curve)
        return res
    
    return tuple([make_swaption(x) for x in df[mask].to_numpy()])      

class Cap:
    
    def __init__(self,parameters:tuple):
        self.vol,self.tenor,self.maturity=parameters[0],parameters[1],parameters[2] #Vol quote en  %
        self.typequote=parameters[3]
        self.frequency=self.tenor
        self.strike=parameters[4]

    def __repr__(self):
        return f'Cap (Tenor:{self.tenor},Maturity:{self.maturity},Strike:{self.strike},quote:{self.vol})'
    
    def compute_mkt_price(self,calc_date:ql.Date,curve:Curve,daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis)):
        
        start_date=calc_date + ql.Period(self.frequency)
        end_date=start_date + ql.Period(self.maturity)
        schedule=list(ql.MakeSchedule(start_date,end_date,ql.Period(self.frequency)))[1:]
        zc=curve.discount_factor(schedule)

        tgrid=np.array([daycount_calendar.yearFraction(calc_date,y) for y in schedule])
        delta=np.array([x-y for x,y in zip(tgrid[1:],tgrid)])

        #compute strike
        if self.strike=='ATM':
            lvl=np.sum(zc[1:]*delta)
            strike=(zc[0]-zc[-1])/lvl
        else:
            strike=float(self.strike)

        ZC_ratio=np.array([(x-y)/y for x,y in zip(zc,zc[1:])])
        L_ratio=ZC_ratio/delta
        #Quote en %.
        vol_=(self.vol*np.sqrt(tgrid[1:]))

         #Retrieve param to compute theorical price
        self.ZC=zc
        self.delta=delta
        self.K=strike
        self.tgrid=tgrid

        if (self.typequote== 'normal_vol'):
            k=(L_ratio-strike)/vol_
            self.mkt_price=self.vol*np.sum(zc[1:]*np.sqrt(tgrid[1:])*delta*(np.exp(-0.5*k**2)/np.sqrt(2*np.pi) + k*norm.cdf(k)))*10000
        
        else:
            d1=(np.log(L_ratio/strike) + 0.5*vol_**2)/vol_
            d2=d1-vol_
            Caplets=L_ratio*norm.cdf(d1) - strike*norm.cdf(d2)
            self.mkt_price=np.sum(zc[1:]*delta*Caplets)*10000
        
def select_and_prepare_caps(df:pd.DataFrame,curve,calc_date:ql.Date,currency:str) ->tuple[Cap]:
    
    mask=df.Description.str.contains(currency and 'Cap')
    
    def make_cap(x):
        name,quote=x[0].split(),x[1]
        t_indicator=[x for x in name if any(y.isdigit()==True for y in x)]
        tenor,maturity=t_indicator[0],t_indicator[1]
        strike=name[-1]
        if 'normal' in name[0]:
            type_quote='normal_vol'
        else :
            type_quote='vol'
        param=(quote,tenor,maturity,type_quote,strike)
        res=Cap(param)
        res.compute_mkt_price(calc_date,curve)
        return res
    
    return tuple([make_cap(x) for x in df[mask].to_numpy()]) 

