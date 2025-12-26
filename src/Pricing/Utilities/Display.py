import pandas as pd
import QuantLib as ql
import numpy as np
from datetime import datetime

def translate_bool(b:bool)->str:
    if b:
        return 'Oui'
    else:
        return 'Non'

def translate_freq(item:str) -> str:
    if item=='A':
        return 'Annuelle'
    if item=='Q':
        return 'Trimestrielle'
    if item=='S':
        return 'Semestrielle'
    if item=='M':
        return 'Mensuelle'

def truncate(n:float,decimals=0) -> int:
    multiplier=10**decimals
    return int(n*multiplier)/multiplier

def format_to_percent(x:float) -> str:
    x=truncate(x*100,2)
    return str(x)+'%'

def format_to_bps(x:float) -> str:
    x=truncate(x*10000,2)
    return str(x)+'bps'

def format_ql_date(date:ql.Date,format:str) -> str:
    date=datetime(date.year(),date.month(),date.dayOfMonth())
    return date.strftime(format)


def display_pricing_results(res:dict):

    print('Price',format_to_percent(res['price']))
    print('Duration',res["duration"])
    if 'coupon' in res.keys():
        print('Coupon',format_to_percent(res["coupon"]))
    print('funding_spread',format_to_bps(res["funding_spread"]))
    if not "funding_table" in res.keys():
        print(pd.DataFrame(res["table"]))
        return
    else:
        print('Structure')
        print(pd.DataFrame(res["table"]))
        print('Funding')
        print(pd.DataFrame(res["funding_table"]))
        return
