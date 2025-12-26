import pandas as pd
import numpy as np
import QuantLib as ql
from scipy.interpolate import CubicSpline

def get_filename_from_date(date:ql.Date,prefix='market_data_') -> str:

    y=str(date.year())
    m=date.month()
    d=date.dayOfMonth()

    if m<10:
        m='0'+str(m)
    else:
        m=str(m)

    if d<10:
        d='0'+str(d) 
    else:
        d=str(d)

    name= y +'-'+ m +'-'+ d
    return prefix + name +'.xlsx'

def select_row_from_keywords(df:pd.DataFrame,col:list[str],keywords:list[str]):
    mask=np.ones(len(df)).astype(bool)
    for word in keywords:
        mask*=df[col].str.contains(word)
    return mask

def concat_df_from_mktdata_file(File:pd.DataFrame,sheet_list:list[str]):

    df_list=[]
    for sheet in sheet_list:
        df=File.parse(sheet_name=sheet).iloc[:,:3]
        df.iloc[:,0]=df.iloc[:,0] + ' ' + df.iloc[:,1]
        df.iloc[:,2]=pd.to_numeric(df.iloc[:,2],errors="coerce")
        df=df.drop(df.columns[1],axis=1)
        df=df.dropna()
        df_list.append(df)
    
    res=pd.concat(df_list)
    res.columns=['Description','Quote']
    return res

def concatenate_df(File:pd.DataFrame,keyword:str,sheet_list:list[str]) -> pd.DataFrame:
    
    df_list=[File.parse(sheet_name=x).dropna() for x in sheet_list]
    for i in range(len(df_list)):
        df_list[i]=df_list[i].loc[df_list[i].iloc[:,1].str.contains(keyword)]
        df_list[i].iloc[:,0]= df_list[i].iloc[:,0] + ' ' + df_list[i].iloc[:,1]        
        mask=[ i for i,x in enumerate(df_list[i].iloc[:,2]) if type(x) not in [float,int] ]
        df_list[i].drop(df_list[i].index[mask],inplace=True)
        
    res=pd.concat(df_list)
    res=res.drop(res.columns[[1,3,4]],axis=1)
    res.columns=['Description','Quote']
    return res

def get_ref_quote(File:pd.ExcelFile,cur_name:str,calc_date:ql.Date):

    df=concat_df_from_mktdata_file(File,sheet_list=('Swaps',))
    mask=select_row_from_keywords(df,"Description",keywords=[cur_name,'Swap_rate'])
    # df=concatenate_df(File,cur_name,sheet_list=('Deposits','Futures','Swaps'))
    #mask=df.Description.str.contains("Basis_swap") & df.Description.str.contains("V_3M")
    df=df[mask]
    ref=df.iloc[:,1].to_numpy()
    def extract_maturity(item):
        if '@' in item:
            return item.split('@')[0].split()[-1]
        else:
            return item.split()[-1] 
    Grid=df.iloc[:,0].apply( extract_maturity).to_numpy()
    cal=ql.Thirty360(ql.Thirty360.BondBasis)
    tgrid=[cal.yearFraction(calc_date,calc_date+ql.Period(x)) for x in Grid]
    sort_idxs,tgrid=zip(*sorted(enumerate(tgrid), key=lambda i: i[1]))

    return CubicSpline(tgrid,[ref[i] for i in sort_idxs])