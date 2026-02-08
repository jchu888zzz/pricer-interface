import sys
from pathlib import Path
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import pandas as pd
import QuantLib as ql
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from Pricing.Rates import GetResults
from Pricing.Rates.Model import HullWhite
from Pricing.Rates.Payoffs import FixedRate 
from Pricing.Utilities.decorators import timer

def compute_result_run(mkt_data:dict,input:dict,max_workers=4)->tuple[dict]:
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
        params = deepcopy(base_params)
        params['maturity'] = str(maturity)
        params['NC']=str(NC)
        dic_prep=module.precomputation(prep_model['calc_date'],prep_model['model'],
                                        params,prep_model['risky_curve'],risky=True)
        
        return f"{maturity}NC{NC}", module.solve_coupon(dic_prep,prep_model['risky_curve'])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures={}
        for mat in range(maturity_min,maturity_max+1):
            for nc in range(NC_min,mat):
                futures[executor.submit(price_single, mat,nc)]=f"{mat}NC{nc}"
        for future in as_completed(futures):
            try:
                description, result = future.result()
                results[description] = result
            except Exception as e:
                description = futures[future]
                results[description] = {'error': str(e)}

    sorted_keys=sorted(results.keys())
    res={"Name":sorted_keys,
        "Coupon":[results[key][0] for key in sorted_keys],
        "Funding":[results[key][1] for key in sorted_keys]}
    
    return res #pd.DataFrame(results).T.sort_index()

input={
"_source_page": "Rate",
"_source_tab": "FixedRate",
"base_param": {
"NC": "1",
"UF": "2.0%",
"currency": "EUR",
"fixing_days_offset": "-10",
"frequency": "Annually",
"in-fine": "false",
"issue_date": "06.01.2026",
"maturity": "10",
"multi-call": "false",
"solving_choice": "Solve coupon",
"structure_type": "Swap",
"yearly_buffer": "0.0%"
},
"param":{"min_maturity":5,
        "max_maturity":10,
        "min_NC":3}
}
#DataPath=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
DataPath=r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
calc_date=ql.Date(11,11,2025)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)

print(compute_result_run(mkt_data,input,max_workers=4))