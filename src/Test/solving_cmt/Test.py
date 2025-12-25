import os
import QuantLib as ql
import pandas as pd
import sys

from pathlib import Path

pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

from Pricing.Utilities import Display,Data_File,Dates
from Pricing.Rates.Model import HullWhiteCMT

from Pricing.Rates.Payoffs.Autocallable import Autocall
from Pricing.Rates.Payoffs.Callable import RangeAccrual,Digit

from contracts import *

def retrieve_data(path_data:str):
    with pd.ExcelFile(path_data) as File:
        df_curve=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                               sheet_list=('Deposits','Futures','Swaps'))
        df_swaption=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                                  sheet_list=('Swaptions',))
        df_caps=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                              sheet_list=('Caps and Floors',))
        df_issuer=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                                sheet_list=['Grille CIC_EUR','Grille CIC_USD'])
        df_cmt=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                             sheet_list=('Deposits','CMT','Swaps'))
        return {"curve":df_curve,"swaption":df_swaption,"caps":df_caps,
                "issuer":df_issuer,"cmt":df_cmt}

DataPath =r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\19_Quant\CMT\snapshot"
calc_date=ql.Date(11,11,2025)
date_formatted=Dates.ql_to_string(calc_date,'%Y-%m-%d')
path_data=os.path.join(DataPath,"mkt_data_"+date_formatted+ ".xlsx")

data=retrieve_data(path_data)

#select test from contracts
test=test_digit_swap

undl=test['param']['underlying1']
prep_model=HullWhiteCMT.get_model(data,undl,calc_date)

PAYOFF_MAPPING={'Autocall':Autocall.Process,
                'Digit':Digit.Process,
                'RangeAccrual':RangeAccrual.Process}

payoff_type=test['_source_tab']
payoff_class=PAYOFF_MAPPING.get(payoff_type)
if not payoff_class:
    raise ValueError(f'{payoff_type} not implemented')
res=payoff_class.solve_coupon(prep_model,test['param'])
Display.display_pricing_results(res)