import os
import QuantLib as ql
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from .Payoffs.Types import Autocall, Digit, FixedRate, MinMax,Tarn,RangeAccrual
from Pricing.Utilities import Dates,Data_File
from .Model import HullWhiteCMT,HullWhite

#Data Preparation
def get_filename(path_folder:str,calc_date:ql.Date,date_format='%Y-%m-%d',prefix="market_data_",extension=".xlsx") -> str:
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

AUTOCALL_MAPPING={'Autocall':Autocall,
                    'Tarn':Tarn}

CALLABLE_MAPPING={'Digit':Digit,
                    'RangeAccrual':RangeAccrual,
                    'FixedRate':FixedRate,
                    'MinMax':MinMax}

def compute_result_rate(mkt_data:dict,input:dict) -> tuple[dict]:
    if 'underlying1' in input['param'].keys():
        prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,input['param']['currency'],
                                    input['param']['underlying1'])
    else:
        prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,
                                    input['param']['currency'],None)
    print(prep_model)
    if input['_source_tab'] in AUTOCALL_MAPPING.keys():
        module=AUTOCALL_MAPPING.get(input['_source_tab'])
        dic_prep=module.precomputation(prep_model['calc_date'],
                                        prep_model['model'],input['param'])
    else:
        module=CALLABLE_MAPPING.get(input['_source_tab'])
        dic_prep=module.precomputation(prep_model['calc_date'],
                                        prep_model['model'],input['param'],
                                        prep_model['risky_curve'],risky=True)
    
    solving_choice=input['param']['solving_choice']
    if solving_choice=="Price":
        res=module.compute_price(dic_prep,prep_model['risky_curve'])
    elif solving_choice=="Solve coupon":
        coupon,spread=module.solve_coupon(dic_prep,prep_model['risky_curve'])
        print(coupon,spread)
        # new_dic_prep['contract'].structure_type="Bond"
        # bond_res=module.compute_price(new_dic_prep,prep_model['risky_curve'])
        
        # res={"uf":swap_res["price"],
        #     "duration":bond_res["duration"],
        #     "funding_spread":spread,
        #     "coupon":coupon,
        #     "table":bond_res["table"]}
        
    else :
        raise ValueError(f"{solving_choice} ,not recognized for this {module}")
    
    return input,res

def compute_result_cmt(mkt_data:dict,input:dict) ->tuple[dict]:
        
    OPTION_MAPPING={"Price":{'Autocall':Autocall.Process.compute_price,
                    'Tarn':Tarn.Process.compute_price,
                    'Digit':Digit.Process.compute_price,
                    'RangeAccrual':RangeAccrual.Process.compute_price,
                    'MinMax':MinMax.Process.compute_price},
                    
                    "Solve coupon":{'Autocall':Autocall.Process.solve_coupon,
                    'Digit':Digit.Process.solve_coupon,
                    'RangeAccrual':RangeAccrual.Process.solve_coupon}
                    }
    
    payoff_type=input['_source_tab']
    PAYOFF_MAPPING=OPTION_MAPPING.get(input['param']['solving_choice'])
    func=PAYOFF_MAPPING.get(payoff_type)
    if not func:
        raise ValueError(f'{func} not implemented')

    prep_model=HullWhiteCMT.get_model(mkt_data['calc_date'],mkt_data,
                                    input['param']['currency'],
                                    input['param']['underlying1'])

    res=func(prep_model,input['param'])
    
    return input,res


def compute_result_run(mkt_data:dict,input:dict,max_workers=4)->tuple[dict]:
    """
    Compute pricing results for multiple parameter combinations using thread pool.
    Properly manages thread lifecycle to avoid hanging threads on application close.
    """
    MODULE_MAPPING={'FixedRate':FixedRate}

    payoff_type=input['_source_tab']
    module=MODULE_MAPPING.get(payoff_type)
    if not module:
        raise ValueError(f'{module} not implemented')

    prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,
                                    input['base_param']['currency'],None)

    results = {}
    base_params=input['base_param']
    maturity_min=input['param']['min_maturity']
    maturity_max=input['param']['max_maturity']
    NC_min=input['param']['min_NC']

    def price_single(maturity:int,NC:int) -> tuple[str,tuple[float,float]]:
        """Price a single contract configuration."""
        params = deepcopy(base_params)
        params['maturity'] = str(maturity)
        params['NC']=str(NC)
        dic_prep=module.precomputation(prep_model['calc_date'],prep_model['model'],
                                        params,prep_model['risky_curve'],risky=True)

        return f"{maturity}NC{NC}", module.solve_coupon(dic_prep,prep_model['risky_curve'])

    # Explicit executor with proper cleanup - avoids hanging threads on app close
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pricing")
    try:
        futures = {}
        for mat in range(maturity_min, maturity_max+1):
            for nc in range(NC_min, mat):
                future = executor.submit(price_single, mat, nc)
                futures[future] = f"{mat}NC{nc}"

        # Collect results with timeout protection
        for future in as_completed(futures, timeout=300):  # 5 minute timeout
            try:
                description, result = future.result(timeout=10)
                results[description] = result
            except Exception as e:
                description = futures[future]
                results[description] = {'error': str(e)}

    finally:
        # Critical: ensure threads are cleaned up before returning
        executor.shutdown(wait=True, timeout=10)

    sorted_keys = sorted(results.keys())
    res = {
        "Name": sorted_keys,
        "Coupon": [results[key][0] if 'error' not in results[key] else None for key in sorted_keys],
        "Funding": [results[key][1] if 'error' not in results[key] else None for key in sorted_keys]
    }

    return res
