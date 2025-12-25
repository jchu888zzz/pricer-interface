import pandas as pd
import QuantLib as ql
import typing
import numpy as np
from scipy.interpolate import CubicSpline

from Pricing.Utilities import Data_File,InputConverter
from Pricing.Curves import Instruments
from Pricing.Curves.Classic import Curve

# def GetCurve(File:pd.ExcelFile,calc_date:ql.Date,tag:str):

#     instruments=get_CMT_instru(File,calc_date,tag)
#     res=CMT_Curve(instruments,'EUR')
#     res.retrieve_interp(File,calc_date,tag)
#     return res

class Curve(Curve):
    def __init__(self,Instruments:list,cur_name:str):
        super().__init__(Instruments,cur_name)

    def retrieve_interp(self,df:pd.DataFrame,calc_date:ql.Date,tag:str):

        mask_cmt=Data_File.select_row_from_keywords(df,'Description',keywords=['EUR',tag])
        if self.cur_name=='EUR':
           keywords= ['EUR','ESTR']
        elif self.cur_name=='USD':
            keywords= ['USD','SOFR']
        else : 
            raise ValueError(f'{self.cur_name} Not implemented')
        mask_ois=Data_File.select_row_from_keywords(df,'Description',keywords=keywords)

        self.interp_cmt=get_interp(df[mask_cmt],calc_date,option='Linear')
        self.interp_ois=get_interp(df[mask_ois],calc_date,option='Cubic')

    def ajusted_fwd_cms_interp(self,Tenor:str,t:float):
        repo_spread=0.001
        T=InputConverter.convert_period(Tenor)
        fwd_adjusted=fwd(self.interp_cmt,self.interp_ois,repo_spread,t,T,cx_adj=0.0005)
        return fwd_adjusted

def get_instru(df:pd.DataFrame,calc_date:ql.Date,tag=str)-> list:

    day_count=ql.Actual360()
    cal=ql.TARGET()

    dic_deposit={'O_N':'1D','T_N':'2D','S_N':'3D'}

    def Func_Deposit(item):
        name=item[0].split()
        quote=item[1]
        if name[-1] in dic_deposit.keys():        
            T_date=cal.advance(calc_date,ql.Period(dic_deposit[name[-1]]))
        else:
            T_date=cal.advance(calc_date,ql.Period(name[-1]))    
        T=day_count.yearFraction(calc_date,T_date)
        return Instruments.Deposit(name[-1],T,quote)

    def Func_swap(item,fixed_freq,float_freq):
        
        name=item[0].split('@')[0].split()
        T_date=cal.advance(calc_date+ql.Period("2D"),ql.Period(name[-1]))
        T=day_count.yearFraction(calc_date,T_date)
        return Instruments.Swap(name[-1],T,item[1],fixed_freq,float_freq)

    mask_deposit=Data_File.select_row_from_keywords(df,'Description',keywords=('EUR','Deposit'))
    mask_swap=Data_File.select_row_from_keywords(df,'Description',keywords=('EUR','Basis_swap',tag))
    
    Deposits=list(map(lambda x: Func_Deposit(x),df[mask_deposit].to_numpy()))
    Swaps=list(map(lambda x: Func_swap(x,'1Y','1Y'),df[mask_swap].to_numpy()))
    Deposits=[ x for x in Deposits if x.T <1]
    res=sorted(Deposits+ Swaps,key=lambda x:x.T)
    return res

def get_interp(df:pd.DataFrame,calc_date:ql.Date,option='Cubic') -> typing.Callable:
    cal=ql.Actual360()
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

