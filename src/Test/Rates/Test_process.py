import sys
from pathlib import Path

from Pricing.Rates.Payoffs import Autocall, Digit, FixedRate, MinMax
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd
import numpy as np

from Pricing.Utilities import Display
from Pricing.Rates import GetResults
from Pricing.Rates.Model import  HullWhite
from Pricing.Rates.Payoffs import TARN
from Pricing.Rates.Payoffs import RangeAccrual

#ÂDataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
DataPath=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
calc_date=ql.Date(6,1,2026)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)

input={
"_source_page": "Rate",
"_source_tab": "FixedRate",
"param": {
"NC": "1",
"UF": "2.0%",
"currency": "EUR",
"fixing_days_offset": "-10",
"frequency": "Annually",
"in-fine": "false",
"issue_date": "06.01.2026",
"maturity": "10",
"multi-call": "true",
"solving_choice": "Solve coupon",
"structure_type": "Bond",
"yearly_buffer": "0.0%"
}
}

currency=input['param']['currency']

if "underlying1" in input["param"].keys():
    undl=input['param']['underlying1']
else:
    undl=None

prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,currency,
                                undl)

AUTOCALL_MAPPING={'Autocall':Autocall.precomputation,
                    'Tarn':TARN.precomputation}

CALLABLE_MAPPING={'Digit':Digit.precomputation,
                    'RangeAccrual':RangeAccrual.precomputation,
                    'FixedRate':FixedRate.precomputation,
                    'MinMax':MinMax.precomputation}

if input['_source_tab'] in ['Autocall','Tarn']:
    dic_prep=AUTOCALL_MAPPING.get(input['_source_tab'])(prep_model['calc_date'],
                                                        prep_model['model'],input['param'])
else:
    dic_prep=CALLABLE_MAPPING.get(input['_source_tab'])(prep_model['calc_date'],
                                                        prep_model['model'],input['param'],
                                                        prep_model['risky_curve'],risky=True)
    
contract=dic_prep['contract']

OPTION_MAPPING={"Price":{'Autocall':Autocall.compute_price,
                    'Tarn':TARN.compute_price,
                    'Digit':Digit.compute_price,
                    'RangeAccrual':RangeAccrual.compute_price,
                    'FixedRate':FixedRate.compute_price,
                    'MinMax':MinMax.compute_price},

                    "Solve coupon":{'Autocall':Autocall.solve_coupon,
                    'Digit':Digit.solve_coupon,
                    'RangeAccrual':RangeAccrual.solve_coupon,
                    'FixedRate':FixedRate.solve_coupon}
                    }

solving_choice=input['param']['solving_choice']
payoff=input["_source_tab"]
res=OPTION_MAPPING[solving_choice][payoff](dic_prep,
                                            prep_model["risky_curve"])

if solving_choice=="Solve coupon":
    dic_prep_new=dic_prep.copy()
    dic_prep_new['contract'].coupon=res[0]
    dic_prep_new['contract'].funding_spread=res[1]
    
    swap_res=OPTION_MAPPING["Price"][payoff](dic_prep,
                                            prep_model["risky_curve"])
    final_result={}
    final_result['price']=swap_res['price']
    final_result['coupon']=res[0]
    final_result['funding']=res[1]
    
print("final",final_result)