import QuantLib as ql
import re

def set_param(item:str,standard:float)-> float:
    
    if item=='':
        return standard
    elif '%' in item:
        temp=item.split('%')
        return float(temp[0])/100
    elif 'bps' in item:
        temp=item.split('bps')
        return float(temp[0])/10000
    else : 
        return float(item)
    
def freq_converter(item:str)-> str:
    dic={'Annually':ql.Period('1Y'),'Semi-annually':ql.Period('6M'),
                    'Quarterly':ql.Period('3M'),'Monthly':ql.Period('1M')}
    return dic[item]

def convert_period(period:str) -> str:
    dic={'D':1/360,'W':7/360,'M':30/360,'Y':1}
    temp=re.split(r'(\D+)',period)
    return float(temp[0])*dic[temp[1]]

def convert_date(date:str,sep='.') -> ql.Date:
    temp=date.split(sep)
    return ql.Date(int(temp[0]),int(temp[1]),int(temp[2]))

def check_option_type(char:str) -> str:

    if char in ['Bond','Swap']:
        return char
    raise ValueError( ' Option type must be Bond or Swap')
