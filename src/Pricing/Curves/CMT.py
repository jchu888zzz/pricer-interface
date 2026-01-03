import pandas as pd
import QuantLib as ql
import typing
import numpy as np
from scipy.interpolate import CubicSpline

from Pricing.Utilities import Data_File,InputConverter
from Pricing.Curves import Classic

class Curve(Classic.Curve):
    def __init__(self,calc_date:ql.Date,cur_name:str,instruments:list):
        super().__init__(calc_date,cur_name,instruments)

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
    
    def forward_cms(self,dates:list[ql.Date],tenor:str='10Y',option:str='adjusted'):
        
        if option=='unadjusted':
            res=np.zeros(len(dates))
            for i,d in enumerate(dates):
                start=d+ql.Period('2D')
                schedule= list(ql.MakeSchedule(start,start+ql.Period(tenor),ql.Period('1Y')))[1:]
                zc=self.discount_factor(schedule)
                res[i]=(zc[0]-zc[-1])/np.sum(zc[1:])
        
            return res
        
        if option=='adjusted':
            repo_spread=0.001
            cx_adj=0.0005
            Tenor=InputConverter.convert_period(tenor)
            tgrid=[self.calendar.yearFraction(self.calc_date,d) for d in dates]    
            res=np.array([fwd(self.interp_cmt,self.interp_ois,repo_spread,t,Tenor,cx_adj)
                            for t in tgrid])
            return res
        
        raise ValueError(f'{option} not implemented')

def sort_and_select_instruments(calc_date:ql.Date,df:pd.DataFrame,tag=str)-> list:

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

    mask_deposit=Data_File.select_row_from_keywords(df,'Description',keywords=('EUR','Deposit'))
    mask_swap=Data_File.select_row_from_keywords(df,'Description',keywords=('EUR','Basis_swap',tag))
    
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

