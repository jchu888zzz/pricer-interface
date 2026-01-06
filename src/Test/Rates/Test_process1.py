import sys
from pathlib import Path
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd
import numpy as np

from Pricing.Utilities import Display
from Pricing.Rates import GetResults
from Pricing.Rates.Model import  HullWhite
from Pricing.Rates.Payoffs.Autocallable import TARN, Autocall
from Pricing.Rates.Payoffs.Callable import Digit, FixedRate,RangeAccrual,MinMax

#ÂDataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
DataPath=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
calc_date=ql.Date(6,1,2026)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)

input={
"_source_page": "Rate",
"_source_tab": "FixedRate",
"param": {
"NC": "3",
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

AUTOCALL_MAPPING={'Autocall':Autocall,
                    'Tarn':TARN}

CALLABLE_MAPPING={'Digit':Digit,
                    'RangeAccrual':RangeAccrual,
                    'FixedRate':FixedRate,
                    'MinMax':MinMax}

if 'underlying1' in input['param'].keys():
    prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,input['param']['currency'],
                                input['param']['underlying1'])
else:
    prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,
                                input['param']['currency'],None)

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
res=module.solve_coupon(dic_prep,prep_model['risky_curve'])
print(res)

test=input.copy()
test['param']['coupon']=str(res[0])
test['param']['funding_spread']=str(res[1])
dic_prep1=module.precomputation(prep_model['calc_date'],
                                    prep_model['model'],test['param'],
                                    prep_model['risky_curve'],risky=True)
res=module.compute_price(dic_prep1,prep_model['risky_curve'])
Display.display_pricing_results(res)