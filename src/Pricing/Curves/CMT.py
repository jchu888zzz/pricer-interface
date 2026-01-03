import pandas as pd
import QuantLib as ql
import typing
import numpy as np
from scipy.interpolate import CubicSpline

from Pricing.Utilities import Data_File,InputConverter
from Pricing.Curves import Classic

def get_curve(calc_date:ql.Date,df:dict,currency:str,tag:str):
    instruments=sort_and_select_instruments(calc_date,df,currency,tag)
    res=Curve(calc_date,currency,instruments)
    res.retrieve_interp(calc_date,df,tag)
    return res


class Curve(Classic.Curve):
    def __init__(self,calc_date:ql.Date,currency:str,instruments:list):
        super().__init__(calc_date,currency,instruments)

    def retrieve_interp(self,calc_date:ql.Date,df:pd.DataFrame,tag:str):

        mask_cmt=Data_File.select_row_from_keywords(df,'Description',keywords=['EUR',tag])
        if self.cur_name=='EUR':
            keywords= ['EUR','ESTR']
        elif self.cur_name=='USD':
            keywords= ['USD','SOFR']
        else : 
            raise ValueError(f'{self.cur_name} Not implemented')
        mask_ois=Data_File.select_row_from_keywords(df,'Description',keywords=keywords)

        self.interp_cmt=get_interp(df[mask_cmt],calc_date,self.calendar,option='Linear')
        self.interp_ois=get_interp(df[mask_ois],calc_date,self.calendar,option='Cubic')

    # def ajusted_fwd_cms_interp(self,t:float,tenor:str='10Y'):
    #     repo_spread=0.001
    #     T=InputConverter.convert_period(tenor)
    #     fwd_adjusted=fwd(self.interp_cmt,self.interp_ois,repo_spread,t,T,cx_adj=0.0005)
    #     return fwd_adjusted
    
    def forward_cms(self,t:float,tenor:str='10Y',option:str='adjusted') ->np.ndarray:
        tenor=InputConverter.convert_period(tenor)
        fixgrid=t + np.arange(0,tenor,1)
        if option=='unadjusted':
            zc=self.discount_factor_from_times(fixgrid)
            res=(zc[0]-zc[-1])/np.sum(zc[1:])
            return res
        
        if option=='adjusted':
            repo_spread=0.001
            cx_adj=0.0005 
            res=fwd(self.interp_cmt,self.interp_ois,repo_spread,t,tenor,cx_adj)              
            return res
        
        raise ValueError(f'{option} not implemented')

def sort_and_select_instruments(calc_date:ql.Date,df:pd.DataFrame,currency:str,tag=str)-> list:

    res=[]

    def make_deposit(item):
        name=item[0].split()
        res=Classic.Deposit(period=name[-1],quote=item[1])
        period=res.convert_period()
        res.maturity_date=Classic._BUSINESS_CALENDAR.advance(calc_date,ql.Period(period),
                                                                ql.ModifiedFollowing)
        return res

    def make_swap(item):
        name=item[0].split('@')[0].split()
        fix_freq='1Y'
        float_freq='1Y'
        res=Classic.Swap(period=name[-1],quote=item[1],fix_freq=fix_freq,float_freq=float_freq)
        res.maturity_date=Classic._BUSINESS_CALENDAR.advance(calc_date+ql.Period("2D"),
                                                            ql.Period(res.period),ql.ModifiedFollowing)
        res.fix_schedule=list(ql.MakeSchedule(calc_date+ql.Period("2D"),
                                            res.maturity_date,ql.Period(fix_freq)))[1:]
        res.float_schedule=list(ql.MakeSchedule(calc_date+ql.Period("2D"),
                                                res.maturity_date,ql.Period(float_freq)))[1:]
        return res

    mask_deposit=Data_File.select_row_from_keywords(df,'Description',keywords=(currency,'Deposit'))
    mask_swap=Data_File.select_row_from_keywords(df,'Description',keywords=(currency,'Basis_swap',tag))
    
    deposits=list(map(make_deposit,df[mask_deposit].to_numpy()))
    if deposits:
        deposits=[ x for x in deposits if x.maturity_date <calc_date+ql.Period('1Y')]
        res.extend(deposits)
    swaps=list(map(make_swap,df[mask_swap].to_numpy()))
    if swaps:
        res.extend(swaps)
    
    return sorted(res,key=lambda x: x.maturity_date)

def get_interp(df:pd.DataFrame,calc_date:ql.Date,cal,option='Cubic') -> typing.Callable:
    df_temp=df.copy()
    df_temp['Maturities']=df_temp['Description'].apply(lambda x: x.split(' ')[-2])
    df_temp['tgrid']=[cal.yearFraction(calc_date,calc_date+ql.Period(x)) for x in df_temp['Maturities']]
    df_temp=df_temp.sort_values(by='tgrid')
    if option=='Linear':
        return lambda x : np.interp(x, df_temp['tgrid'].values, df_temp['Quote'].values )
    elif option=='Cubic':
        return CubicSpline(df_temp['tgrid'].values, df_temp['Quote'].values)
    else:
        raise ValueError('Not recognized')

def DV01(x:float,n=10) -> float:
    return sum([(1+x)**(-i) for i in range(n+1)])
    
def fwd(interp_yield,interp_ois,repo_spread,t,T,cx_adj=0.0005):
    y=interp_yield(t+T)
    return y + t/DV01(y)*(y-(repo_spread+interp_ois(t))) +cx_adj

