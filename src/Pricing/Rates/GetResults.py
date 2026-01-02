import os
import QuantLib as ql
from Pricing.Utilities import Dates,Data_File
import pandas as pd

from .Payoffs.Autocallable import TARN, Autocall
from .Payoffs.Callable import Digit, FixedRate,RangeAccrual,MinMax
from .Model import HullWhiteCMT,HullWhite

#Data Preparation
def get_filename(path_folder:str,calc_date:ql.Date,date_format='%Y-%m-%d',prefix="mkt_data_",extension=".xlsx") -> str:
    date_formatted=Dates.ql_to_string(calc_date,date_format)
    return os.path.join(path_folder,prefix+date_formatted+extension)

def retrieve_data(path_folder:str,date:ql.Date) -> dict[pd.DataFrame]:

    path=get_filename(path_folder,date)
    with pd.ExcelFile(path) as File:
        df_curve=Data_File.concat_df_from_mktdata_file(File,
                                                    sheet_list=('Deposits','Futures','Swaps'))

        df_swaption=Data_File.concat_df_from_mktdata_file(File,
                                                        sheet_list=('Swaptions',))
        df_caps=Data_File.concat_df_from_mktdata_file(File,
                                                    sheet_list=('Caps and Floors',))
        df_issuer=Data_File.concat_df_from_mktdata_file(File,
                                                    sheet_list=['Grille CIC_EUR','Grille CIC_USD'])
        df_cmt=Data_File.concat_df_from_mktdata_file(File,
                                                    sheet_list=('Deposits','CMT','Swaps'))
        
    return {"curve":df_curve,
        "swaption":df_swaption,
        "caps":df_caps,
        "cmt":df_cmt,
        "issuer":df_issuer,
        'calc_date':date}


def compute_result_rate(mkt_data:dict,input:dict) -> tuple[dict]:
    OPTION_MAPPING={"Price":{'Autocall':Autocall.Process.compute_price,
                    'Tarn':TARN.Process.compute_price,
                    'Digit':Digit.Process.compute_price,
                    'RangeAccrual':RangeAccrual.Process.compute_price,
                    'FixedRate':FixedRate.Process.compute_price,
                    'MinMax':MinMax.Process.compute_price},

                    "Solve coupon":{'Autocall':Autocall.Process.solve_coupon,
                    'Digit':Digit.Process.solve_coupon,
                    'RangeAccrual':RangeAccrual.Process.solve_coupon,
                    'FixedRate':FixedRate.Process.solve_coupon}
                    }
    
    payoff_type=input['_source_tab']
    PAYOFF_MAPPING=OPTION_MAPPING.get(input['param']['solving_choice'])
    func=PAYOFF_MAPPING.get(payoff_type)
    if not func:
        raise ValueError(f'{func} not implemented')

    prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,input['param']['currency'])
    res=func(prep_model,input['param'])
    return input,res

def compute_result_cmt(mkt_data:dict,input:dict) ->tuple[dict]:
        
    OPTION_MAPPING={"Price":{'Autocall':Autocall.Process.compute_price,
                    'Tarn':TARN.Process.compute_price,
                    'Digit':Digit.Process.compute_price,
                    'RangeAccrual':RangeAccrual.Process.compute_price,
                    'MinMax':MinMax.MinMax.compute_price,
                    'FixedRate':MinMax.MinMax.compute_price},
                    

                    "Solve coupon":{'Autocall':Autocall.Process.solve_coupon,
                    'Digit':Digit.Process.solve_coupon,
                    'RangeAccrual':RangeAccrual.Process.solve_coupon}
                    }
    
    payoff_type=input['_source_tab']
    PAYOFF_MAPPING=OPTION_MAPPING.get(input['param']['solving_choice'])
    func=PAYOFF_MAPPING.get(payoff_type)
    if not func:
        raise ValueError(f'{func} not implemented')

    prep_model=HullWhiteCMT.get_model(mkt_data,input['param']['underlying1'],
                                    mkt_data['calc_date'])
    res=func(prep_model,input['param'])
    
    return input,res
