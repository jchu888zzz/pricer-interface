import QuantLib as ql
import pandas as pd
import sys

from pathlib import Path

pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

from Pricing.Rates import GetResults
from Pricing.Utilities import Display
from Pricing.Rates.Model import HullWhite
from Pricing.Rates.Payoffs.Autocallable import TARN, Autocall
from Pricing.Rates.Payoffs.Callable import Digit, FixedRate,RangeAccrual,MinMax

from contracts import *

DataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
calc_date=ql.Date(11,11,2025)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)
#select test from contracts
input=test_autocall_swap

prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,input['param']['currency'])


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


