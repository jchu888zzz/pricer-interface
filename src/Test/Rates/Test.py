import sys
from pathlib import Path
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd
import numpy as np

from Pricing.Rates import GetResults
from Pricing.Utilities import Display,Functions
from Pricing.Curves import Classic


DataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
calc_date=ql.Date(11,11,2025)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)
currency='EUR'
curve,risky_curve=Classic.get_curves(calc_date,mkt_data,currency)


# def instantaneous_f(t,h=0.01):
#     res=-(np.log(curve.discount_factor_from_times(t+h))-np.log(curve.discount_factor_from_times(t)))/h
#     return res

# grid=np.arange(1.8,2.3,0.05)
# temp=Functions.integral_cst_by_part(curve.rates, curve.tgrid, grid, eps=0.1)
# print(pd.DataFrame({'grid':curve.tgrid,'value':curve.rates}))
# print(temp)
#dates=list(ql.MakeSchedule(calc_date,calc_date+ql.Period("50Y"),ql.Period('1Y')))
#print(pd.DataFrame({'dates':dates,'value':curve.discount_factor(dates)}))



# import numpy as np
# dates=list(ql.MakeSchedule(calc_date,calc_date+ql.Period("10Y"),ql.Period('1Y')))

# print(risky_curve.model.compute_default_proba(np.arange(0,10,1)))


from Pricing.Rates.Model import  HullWhite
from Pricing.Rates import Instruments

option='swaption'
prep_model=HullWhite.get_model(calc_date,mkt_data,currency,option)
model=prep_model['model']
if option=='swaption':
    instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                        curve,calc_date,currency)
    instruments=[x for x in instruments if x.strike_type=='ATM']
elif option=="cap":
    instruments=Instruments.select_and_prepare_caps(mkt_data['caps'],curve,calc_date,currency)

df=pd.DataFrame({'Item':instruments,
                    'Mkt price':[x.mkt_price for x in instruments],
                    'Th price':[model.price_swaption(x) for x in instruments]})
print(df)
# input={'_source_tab':'FixedRate',
#                 'param':{'issue_date':'30.11.2025',
#                         'maturity':'8',
#                         'fixing_days_offset':'-5',
#                         'frequency':'Annually',
#                         'coupon':'4%',
#                         'fixing_type':'in arrears',
#                         'currency':'EUR',
#                         'structure_type':'Swap',
#                         'funding_spread':'90bps',
#                         'solving_choice':'Price'}}
# data_callable={'multi-call': 'true',
#                 'NC':'3'}
# input['param'].update(data_callable)

# from Pricing.Rates.Payoffs.Callable import FixedRate 
# from Pricing.Rates import Funding
# from Pricing.Utilities import Functions
# import numpy as np

# contract=FixedRate.FixedRate(input['param'])
# data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
#                        cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=0)

# cal=ql.Actual360()

# leg_funding=Funding.Leg(contract,contract.currency)
# rates,schedule=data_rates['rates'],data_rates['schedule'][1:]
# idxs=Functions.find_idx(schedule,leg_funding.fix_dates)