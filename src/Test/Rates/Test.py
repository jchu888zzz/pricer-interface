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
option='cap'
if option=='swaption':
    instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                        curve,calc_date,currency)
    instruments=[x for x in instruments if x.strike_type=='ATM']
elif option=="cap":
    instruments=Instruments.select_and_prepare_caps(mkt_data['caps'],curve,calc_date,currency)

# print(pd.DataFrame({'Item':instruments,
#                     'Mkt price':[x.mkt_price for x in instruments],
#                     'Th price':[model.price_swaption(x) for x in instruments]}))

