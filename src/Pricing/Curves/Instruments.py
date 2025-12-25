
import pandas as pd
import QuantLib as ql

class Deposit:
    
    def __init__(self,period:str,T:float,quote:float):
        
        self.period=period
        self.T,self.quote=T,quote
    
    def __repr__(self):
        
        return f'Deposit (Period:{self.period},quote:{self.quote})'

class Future:
    
    def __init__(self,Delivery:str,T:float,quote:float):
        
        self.Delivery=Delivery
        self.T,self.quote=T,quote

    def __repr__(self):
        
        return f'Futures (Delivery:{self.Delivery},quote:{self.quote})'

class Swap:
    
    def __init__(self,period:str,T:float,quote:float,fixed_freq:str,float_freq:str):
        
        self.period=period
        self.T,self.quote=T,quote
        self.fix_delta,self.float_delta=fixed_freq,float_freq
    
    def __repr__(self):
        
        return f'Swap (Period:{self.period},quote:{self.quote})'
    

def get_third_wednesday(month:int,year:int):
    
    t=ql.Date(15,month,year)
    if t.weekday()!=4:
        day= 15 + (4-t.weekday())%7
        
        return ql.Date(day,month,year)
    else:
        
        return t

def get_instruments(df:pd.DataFrame,calc_date:ql.Date,cur_name:str,option='3M') -> list  :
    """Retrieve Curve Instruments from a formatted dataframe
       Returns:
        sorted list of Instruments  Deposits/Future/Swaps """

    df=df[~df.Description.str.contains("@BFR")]

    day_count=ql.Actual360()
    months={'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    dic_deposit={'O_N':'1D','T_N':'2D','S_N':'3D'}

    dic_euro={'Overnight':('1Y','1Y'),'1M':('1Y','1M'),'3M':('1Y','3M'),'6M':('1Y','6M'),'Classical':('1Y','6M') }
    dic_usd={'Overnight':('1Y','1Y'),'1M':('3M','3M'),'3M':('6M','3M'),'6M':('6M','6M'), 'Classical':('1Y','6M')}
    dic_freq_swap={"EUR":dic_euro,'USD':dic_usd}
    cal=ql.TARGET()
    
    ref={'EUR': (0.75,1.9),'USD':(0.48,1), 'JPY':(0.48,1.75),'GBP':(0.75,1.75),'CHF':(0.75,1.25),'HKD':(0.48,0.75) }
    
    def Func_Deposit(item):
        name=item[0].split()
        if name[-1] in dic_deposit.keys():        
            T_date=cal.advance(calc_date,ql.Period(dic_deposit[name[-1]]))
        else:
            T_date=cal.advance(calc_date,ql.Period(name[-1]))    
        T=day_count.yearFraction(calc_date,T_date)
        return Deposit(name[-1],T,item[1])
    
    def Func_Swap(item,fixed_freq,float_freq):
        
        if '@' in item[0]:
            name=item[0].split('@')[0].split()
        else:
            name=item[0].split()
        T_date=cal.advance(calc_date+ql.Period("2D"),ql.Period(name[-1]))
        T=day_count.yearFraction(calc_date,T_date)
        return Swap(name[-1],T,item[1],fixed_freq,float_freq)
    
    def Func_Basis_Swap(item,fixed_freq,float_freq):
        if '@' in item[0]:
            name=item[0].split('@')[0].split()
        else:
            name=item[0].split()
        
        T_date=cal.advance(calc_date+ql.Period("2D"),ql.Period(name[-1]))
        T=day_count.yearFraction(calc_date,T_date)
        return Swap(name[-1],T,item[1],fixed_freq,float_freq)
    
    def Func_Future(item):
        name=item[0].split()
        if '@' in item[0]:
            month,year=name[-3],int(name[-2])
        else:
            month,year=name[-2],int(name[-1])
        start=get_third_wednesday(months[month],year)
        T_date=cal.advance(start,ql.Period('3M'),ql.ModifiedFollowing)
        T=day_count.yearFraction(calc_date,T_date)
        return Future(month +str(year),T,item[1])
    
    fixed_freq,float_freq=dic_freq_swap[cur_name][option]

    if option=="Overnight":
        mask_deposit=df.Description.str.contains('Deposit')
        deposit_item=list(map(Func_Deposit,df[mask_deposit].to_numpy()))
        deposit_item=sorted(deposit_item,key=lambda x:x.T)
        
        mask_swap=df.Description.str.contains(r'^(?=.*Basis_swap)(?=.*V_Overnight)')
        swap_item=list(map(lambda x: Func_Swap(x,fixed_freq,float_freq),df[mask_swap].to_numpy()))
        swap_item=sorted(swap_item,key=lambda x:x.T)
        
        First_swap_tenor=swap_item[0].T
        deposit_item=[x for x in deposit_item if x.T < First_swap_tenor]
        
        return deposit_item+swap_item
        
    if option=='3M':
        bounds=ref[cur_name]
        
        mask_deposit=df.Description.str.contains(r'^(?=.*Deposit)(?=.*3M)')
        deposit_item=list(map(Func_Deposit,df[mask_deposit].to_numpy()))
        deposit_item=[x for x in deposit_item if x.T < bounds[0] ]
        
        mask_future=df.Description.str.contains(r'^(?=.*futures)(?=.*3M)')
        futures_item=list(map(Func_Future,df[mask_future].to_numpy()))
        
        futures_item=[x for x in futures_item if (x.T > bounds[0] and x.T <= bounds[1])]
        futures_item=sorted(futures_item,key=lambda x:x.T)
        
        mask_swap=df.Description.str.contains(r'^(?=.*Basis_swap)(?=.*V_3M)')
        swap_item=list(map(lambda x: Func_Swap(x,fixed_freq,float_freq),df[mask_swap].to_numpy()))
        swap_item=[x for x in swap_item if x.T > bounds[1]]
        swap_item=sorted(swap_item,key=lambda x:x.T)
        
        return deposit_item+futures_item+swap_item
    
    if option=='Classical':
        bounds=ref[cur_name]
        
        mask_deposit=df.Description.str.contains('Deposit')
        deposit_item=list(map(Func_Deposit,df[mask_deposit].to_numpy()))
        deposit_item=[x for x in deposit_item if x.T < bounds[0] ]
        deposit_item=sorted(deposit_item,key=lambda x:x.T)
        
        mask_future=df.Description.str.contains(r'^(?=.*futures)(?=.*3M)')
        futures_item=list(map(Func_Future,df[mask_future].to_numpy()))
        
        futures_item=[x for x in futures_item if (x.T > bounds[0] and x.T <= bounds[1])]
        futures_item=sorted(futures_item,key=lambda x:x.T)
        
        mask_swap=df.Description.str.contains('Swap')
        swap_item=list(map(lambda x: Func_Swap(x,fixed_freq,float_freq),df[mask_swap].to_numpy()))
        swap_item=[x for x in swap_item if x.T > bounds[1]]
        swap_item=sorted(swap_item,key=lambda x:x.T)
        
        return deposit_item+futures_item+swap_item
    
    if option=='6M':
        bounds=ref[cur_name]
        
        mask_deposit=df.Description.str.contains(r'^(?=.*Deposit)(?=.*6M)')
        deposit_item=list(map(Func_Deposit,df[mask_deposit].to_numpy()))
        
        mask_swap=df.Description.str.contains('Swap')
        #mask_swap=df.Description.str.contains(r'^(?=.*Basis_swap)(?=.*V_6M)')
        swap_item=list(map(lambda x: Func_Swap(x,fixed_freq,float_freq),df[mask_swap].to_numpy()))
        swap_item=sorted(swap_item,key=lambda x:x.T)
        
        return deposit_item+swap_item
    
    if option=='1M':
        mask_deposit=df.Description.str.contains(r'^(?=.*Deposit)(?=.*1M)')
        deposit_item=list(map(Func_Deposit,df[mask_deposit].to_numpy()))
        
        mask_swap=df.Description.str.contains(r'^(?=.*Basis_swap)(?=.*V_1M)')
        swap_item=list(map(lambda x: Func_Swap(x,fixed_freq,float_freq),df[mask_swap].to_numpy()))
        swap_item=sorted(swap_item,key=lambda x:x.T)
        
        return deposit_item+swap_item