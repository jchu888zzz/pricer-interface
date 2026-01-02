import QuantLib as ql
import pandas as pd
import sys

from pathlib import Path

pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

from Pricing.Rates import GetResults
from Pricing.Utilities import Display
from Pricing.Curves import Classic

DataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
calc_date=ql.Date(11,11,2025)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)
currency='EUR'
curve,risky_curve=Classic.get_curves(calc_date,mkt_data,currency)

from Pricing.Rates.Model import  HullWhite
from Pricing.Rates import Instruments

prep_model=HullWhite.get_model(calc_date,mkt_data,currency,option='swaption')
model=prep_model['model']
option='swaption'
if option=='swaption':
    instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                        curve,calc_date,currency)
    instruments=[x for x in instruments if x.strike_type=='ATM']
elif option=="cap":
    instruments=Instruments.select_and_prepare_caps(mkt_data['caps'],curve,calc_date,currency)

# print(pd.DataFrame({'Item':instruments,
#                     'Mkt price':[x.mkt_price for x in instruments],
#                     'Th price':[model.price_swaption(x) for x in instruments]}))
# print(model)
# item=instruments[0]
# print(item)
# print(item.K)


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

# print(model)
# grid=np.arange(0,10,1)
# print([model.var_(t) for t in grid])


# fixgrid=np.array([max(0,cal.yearFraction(calc_date,x)) for x in leg_funding.fix_dates ])

# print(np.mean(rates[:,idxs],axis=0))

# fwds=model.compute_deposit_from_rates(rates[:,idxs],fixgrid,'3M')
# fwds1=np.array([model.compute_deposit_from_rates(rates[:,i],t,'3M') for i,t in zip(idxs,fixgrid) ])

# test=np.mean(fwds,axis=0)
# test1=np.mean(fwds1,axis=1).ravel()

# print(pd.DataFrame({'grid':fixgrid,'1':test,'2':test1}))