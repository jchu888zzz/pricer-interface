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

    print('Price',format_to_percent(res['Price']))
    print('Duration',res["duration"])
    if 'coupon' in res.keys():
        print('Coupon',format_to_percent(res["coupon"]))
    print('funding_spread',format_to_bps(res["funding_spread"]))
    if not "Funding" in res.keys():
        keys_to_include=['Payment Dates','Early Redemption Proba','Cash Flows','Zero Coupon']
        if 'Model Forward' in res.keys():
            keys_to_include.insert(1,'Model Forward')
        print(pd.DataFrame({key:res[key] for key in keys_to_include}))
        return
    else:
        print('Structure')
        structure_keys=['Payment Dates','Early Redemption Proba','Cash Flows','Zero Coupon']
        if 'Model Forward' in res.keys():
            structure_keys.insert(1,'Model Forward')
        print(pd.DataFrame({key:res['Structure'][key] for key in structure_keys}))
        print('Funding')
        funding_keys=['Payment Dates','Model Forward','Proba','Cash Flows','Zero Coupon']
        print(pd.DataFrame({key:res['Funding'][key] for key in funding_keys}))
        return
