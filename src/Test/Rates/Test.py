import sys
from pathlib import Path

from Pricing.Rates.Payoffs import Autocall, Digit, FixedRate, MinMax
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd
import numpy as np

from Pricing.Rates import GetResults
from Pricing.Utilities import Display,Functions
from Pricing.Curves import Classic
from Pricing.Rates.Model import  HullWhite
from Pricing.Rates import Instruments
from Pricing.Rates.Payoffs import TARN
from Pricing.Rates.Payoffs import RangeAccrual

DataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
calc_date=ql.Date(11,11,2025)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)

input={'_source_tab':'Autocall',
        'param':{'issue_date':'30.11.2025',
        'maturity':'10',
        'fixing_days_offset':'-5',
        'frequency':'Annually',
        'coupon_level':'3.5%',
        'coupon':'3%',
        'autocall_level':'2.25%',
        'underlying1':'EUR CMS 10Y',
        'memory_effect':'false',
        'fixing_type':'in arrears',
        'NC':'1',
        'currency':'EUR',
        'structure_type':'Bond',
        'solving_choice':'Price'}}

currency=input['param']['currency']
option='swaption'
prep_model=HullWhite.get_model(mkt_data['calc_date'],mkt_data,input['param']['underlying1'])
model=prep_model['model']
curve=prep_model['curve']

currency,rate_type,tenor=input['param']['underlying1'].split()
if rate_type=='CMS':
    instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                            curve,calc_date,currency)
    instruments=[x for x in instruments if x.strike_type=='ATM' and x.tenor==tenor]
elif rate_type=='Euribor':
    instruments=Instruments.select_and_prepare_caps(mkt_data['caps'],curve,calc_date,currency)

t_array=np.arange(1,10,1)
print(model.cvx_adj_helper.compute_adjustment(t_array,tenor='10Y'))
# df=pd.DataFrame({'Item':instruments,
#                     'Mkt price':[x.mkt_price for x in instruments],
#                     'Th price':[model.price_swaption(x) for x in instruments]})
# AUTOCALL_MAPPING={'Autocall':Autocall.precomputation,
#                     'Tarn':TARN.precomputation}

# CALLABLE_MAPPING={'Digit':Digit.precomputation,
#                     'RangeAccrual':RangeAccrual.precomputation,
#                     'FixedRate':FixedRate.precomputation,
#                     'MinMax':MinMax.precomputation}

# if input['_source_tab'] in ['Autocall','Tarn']:
#     dic_prep=AUTOCALL_MAPPING.get(input['_source_tab'])(prep_model['calc_date'],
#                                                         prep_model['model'],input['param'])
# else:
#     dic_prep=CALLABLE_MAPPING.get(input['_source_tab'])(prep_model['calc_date'],
#                                                         prep_model['model'],input['param'],
#                                                         prep_model['risky_curve'],risky=True)
    

# contract=dic_prep['contract']

# # fix_dates=contract.fix_dates
# # fixgrid=[curve.calendar.yearFraction(calc_date,d) for d in fix_dates]
# # print(fixgrid)
# # print( [model.instantaneous_f(t,h=0.1) for t in fixgrid])

# # print([model.alpha_T(t,10) for t in fixgrid ])

# print(pd.DataFrame({'grid':curve.tgrid,'rate':curve.rates,'df':curve.value}))

# print(curve.forward_swap_rate(contract.fix_dates[0],tenor='10Y',
#                             fix_freq='1Y',float_freq='6M') )