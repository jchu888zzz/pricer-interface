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
        if self.strike_type=='from_ATM':
            self.strike_shift=parameters[7]
        
    def __repr__(self):
        return f'Swaption (Expiry:{self.expiry},Tenor:{self.tenor},striketype:{self.strike_type + self.strike_shift},quote:{self.vol})'
    
    def get_strike(self,striketype,shift):
        LVL=np.sum(self.ZC[1:]*self.delta)
        ATM=(self.ZC[0]-self.ZC[-1])/LVL
        if striketype=='ATM':
            K=ATM
        if striketype=='from_ATM':
            K=ATM+float(shift)
        return K
    
    def get_timegrid(self,calc_date:ql.Date):
        
        start_date=calc_date + ql.Period(self.Expiry)
        end_date=start_date + ql.Period(self.Term)
        schedule=ql.MakeSchedule(start_date,end_date,self.frequency)
        dates=[dt for dt in schedule]
        delta=[ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(calc_date,y) for y in dates]
        
        return np.array(delta)
    
    
    def price_mkt(self,calc_date:ql.Date,curve:Curve,calendar=ql.Thirty360(ql.Thirty360.BondBasis)):
        
        start_date=calc_date + ql.Period(self.Expiry)
        end_date=start_date + ql.Period(self.Term)
        schedule=list(ql.MakeSchedule(start_date,end_date,ql.Period(self.frequency)))
        #Compute strike
        K=curve.forward_swap_rate(self,[start_date],self.tenor,
                            self.fix_freq,self.float_freq)
        
        if self.strike_type=='from_ATM':
            K+=float(self.strike_shift)
        
        fwd=curve.forward_swap_rate([start_date],self.tenor)
        
        t=calendar.yearFraction(calc_date,start_date)
        vol_=self.vol*np.sqrt(t)
        
        if (self.typequote== 'normal_vol'):
            k=(fwd-self.K)/vol_    
            res=np.sum(self.ZC[1:]*self.delta)*vol_*(np.exp(-0.5*k**2)/np.sqrt(2*np.pi)  + k*norm.cdf(k))*10000
        else:
            d1,d2=0.5*vol_,-0.5*vol_            
            res=np.sum(self.ZC[1:]*self.delta)*self.K*(norm.cdf(d1)-norm.cdf(d2))*10000
        return res

class Cap:
    
    def __init__(self,parameters:tuple,Curve,calc_date:ql.Date):
        self.vol,self.Tenor,self.Maturity=parameters[0],parameters[1],parameters[2] #Vol quote en  %
        self.typequote=parameters[3]
        self.frequency=ql.Period(self.Tenor)
        self.timegrid=self.get_timegrid(calc_date)
        self.T=ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(calc_date,calc_date+ql.Period(self.Maturity))
        self.Forward_Swap_rate=Curve.Forward_Swap_rate
        self.ZC,self.delta=Curve.Discount_Factor(self.timegrid),np.array([x-y for x,y in zip(self.timegrid[1:],self.timegrid)])
        self.K=self.get_strike(parameters[4])
        self.price=self.price_mkt()

    def __repr__(self):
        
        return f'Cap (Tenor:{self.Tenor},T:{self.T},Strike:{truncate(self.K,3)},quote:{self.vol},price:{truncate(self.price,2)})'
    
    def get_strike(self,strike:str):
        if strike=='ATM':
            LVL=np.sum(self.ZC[1:]*self.delta)
            K=(self.ZC[0]-self.ZC[-1])/LVL
        else:
            K=float(strike)
        return K
    
    def get_timegrid(self,calc_date:ql.Date):
        start_date=calc_date + self.frequency
        end_date=start_date + ql.Period(self.Maturity)
        schedule=ql.MakeSchedule(start_date,end_date,self.frequency)
        dates=[dt for dt in schedule]
        delta=[ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(calc_date,y) for y in dates[:-1]]
        return np.array(delta)
    
    def price_mkt(self):
        
        ZC_ratio=np.array([(x-y)/y for x,y in zip(self.ZC,self.ZC[1:])])
        L=ZC_ratio/self.delta
        
        if (self.typequote== 'normal_vol'):
            k=(L-self.K)/(self.vol*np.sqrt(self.timegrid[1:]))
            res=self.vol*np.sum(self.ZC[1:]*np.sqrt(self.timegrid[1:])*self.delta*(np.exp(-0.5*k**2)/np.sqrt(2*np.pi) + k*norm.cdf(k)))*10000
        
        else:
            vol_=self.vol*np.sqrt(self.timegrid[1:]) #  Quote en %.
            d1,d2=(np.log(L/self.K) + 0.5*vol_**2)/vol_,(np.log(L/self.K) -0.5*vol_**2)/vol_
            Caplets=L*norm.cdf(d1) - self.K*norm.cdf(d2)
            res=np.sum(self.ZC[1:]*self.delta*Caplets)*10000
        
        return res
    
class Straddle:

    def __init__(self,parameters,Curve,calc_date):
        self.vol,self.Expiry,self.Term=parameters[0],parameters[1],parameters[2] #Vol quote en  %
        self.typequote=parameters[3]
        self.frequency=ql.Period(parameters[4])
        self.timegrid=self.get_timegrid(calc_date)
        self.t=ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(calc_date,calc_date+ql.Period(self.Expiry))
        self.T=ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(calc_date,calc_date+ql.Period(self.Term))
        self.Forward_Swap_rate=Curve.Forward_Swap_rate
        self.ZC,self.delta=Curve.Discount_Factor(self.timegrid),np.array([x-y for x,y in zip(self.timegrid[1:],self.timegrid)])
        self.striketype=parameters[5]
        self.K=self.get_strike(parameters[5],parameters[6])
        self.price=self.price_mkt()
    
    def __repr__(self):
        return f'Straddle (Expiry:{self.Expiry},T:{self.T},Strike:{truncate(self.K,3)},quote:{self.vol},price:{truncate(self.price,2)})'

    def get_strike(self,striketype,shift):
        LVL=np.sum(self.ZC[1:]*self.delta)
        ATM=(self.ZC[0]-self.ZC[-1])/LVL
        if striketype=='ATM':
            K=ATM
        if striketype=='from_ATM':
            K=ATM+float(shift)
        return K
    
    def get_timegrid(self,calc_date):
        
        start_date=calc_date + ql.Period(self.Expiry)
        end_date=start_date + ql.Period(self.Term)
        schedule=ql.MakeSchedule(start_date,end_date,self.frequency)
        dates=[dt for dt in schedule]
        delta=[ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(calc_date,y) for y in dates]
        return np.array(delta)
    
    def price_mkt(self):
        
        f=self.Forward_Swap_rate(self.t,self.T)
        vol_=self.vol*np.sqrt(self.t)
        
        if (self.typequote== 'normal_vol'):
            k=(f-self.K)/vol_    
            call=np.sum(self.ZC[1:]*self.delta)*vol_*(np.exp(-0.5*k**2)/np.sqrt(2*np.pi)  + k*norm.cdf(k))*10000
            put=np.sum(self.ZC[1:]*self.delta)*vol_*(np.exp(-0.5*k**2)/np.sqrt(2*np.pi)  - k*norm.cdf(-k))*10000
        else:
            d1,d2=0.5*vol_,-0.5*vol_            
            call=np.sum(self.ZC[1:]*self.delta)*self.K*(norm.cdf(d1)-norm.cdf(d2))*10000
            put=np.sum(self.ZC[1:]*self.delta)*self.K*(norm.cdf(-d2)-norm.cdf(-d1))*10000
        
        return call+put

#Récupérer les quotes à partir d'une chaîne de caractère pour les caps et swaptions
def get_swaptions(df:pd.DataFrame,Curve,calc_date:ql.Date,currency:str) -> tuple[Swaption]:
    
    mask=df.Description.str.contains(currency and 'Swaption')
    dic_freq={'USD':{'fix_freq':'6M','float_leg':'3M'},
                'EUR':{'fix_freq':'1Y','float_leg':'6M'}}

    def convert_element(x):
        name,quote=x[0].split(),x[1]
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
        param=(quote,expiry,maturity,type_quote,dic_freq[currency]['fix_freq'],
                dic_freq[currency]['float_freq'],striketype,strike_shift)

        return Swaption(param,Curve,calc_date)
    
    return tuple([convert_element(x) for x in df[mask].to_numpy()]) 
        
def get_caps(df:pd.DataFrame,Curve,calc_date:ql.Date,currency:str) ->tuple[Cap]:
    
    mask=df.Description.str.contains(currency and 'Cap')
    
    def convert_element(x):
        name,quote=x[0].split(),x[1]
        t_indicator=[x for x in name if any(y.isdigit()==True for y in x)]
        tenor,maturity=t_indicator[0],t_indicator[1]
        strike=name[-1]
        if 'normal' in name[0]:
            type_quote='normal_vol'
        else :
            type_quote='vol'
    
        param=(quote,tenor,maturity,type_quote,strike)
        return Cap(param,Curve,calc_date)
    
    return tuple([convert_element(x) for x in df[mask].to_numpy()]) 

#Groupby Functions
def groupby_maturity(items,key):
    dic_type={'Cap':Cap,'Swaption':Swaption}
    Instruments_=[x for x in items if type(x)==dic_type[key]]
    Maturities=set([x.T for x in Instruments_])
    res={k:[] for k in Maturities}
    
    for x in Instruments_:
        key=x.T
        res[key].append(x)
    
    return res

def groupby_expiry(items,key):
    dic_type={'Cap':Cap,'Swaption':Swaption,'SpreadOption':CMSSpreadOption}
    items=[x for x in items if type(x)==dic_type[key]]
    Maturities=set([x.Expiry for x in items])
    res={k:[] for k in Maturities}
    
    for x in items:
        key=x.Expiry
        res[key].append(x)
    
    return res
