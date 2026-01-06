import sys
from pathlib import Path
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd
import numpy as np

from Pricing.Rates import GetResults
from Pricing.Curves import Classic
from Pricing.Rates.Model import  HullWhite
from Pricing.Rates import Instruments


#ÂDataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
DataPath=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
calc_date=ql.Date(2,1,2026)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)

input={
"_source_page": "Rate",
"_source_tab": "Tarn",
"param": {
"coupon": "0.0%",
"coupon_level": "0.0%",
"currency": "EUR",
"fixing_days_offset": "-5",
"fixing_type": "in arrears",
"frequency": "Annually",
"guaranteed_coupon": "4.0%",
"in-fine": "false",
"issue_date": "05.01.2026",
"maturity": "10",
"nb_guaranteed_coupon": "2",
"solving_choice": "Price",
"structure_type": "Bond",
"target": "12.0%",
"underlying1": "EUR CMS 10Y"
}
}

currency=input['param']['currency']
curve,risky_curve=Classic.get_curves(calc_date,mkt_data,currency,'Classical')
if  "underlying1" in input["param"].keys():
    _,rate_type,tenor=input["param"]["underlying1"].split()
    if rate_type=='CMS':
        instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                        curve,calc_date,currency)
        instruments=[x for x in instruments if x.strike_type=='ATM' and x.tenor==tenor]
    elif rate_type=='Euribor':
        instruments=Instruments.select_and_prepare_caps(mkt_data['caps'],curve,calc_date,currency)
    else:
        raise ValueError(f"not implemented")
else:
    instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                        curve,calc_date,currency)
    instruments=[x for x in instruments if x.strike_type=='ATM']
model=HullWhite.Calibration(curve,instruments)
print(model)
#Swaptions
df=pd.DataFrame({'Item':instruments,
                'Mkt price':[x.mkt_price for x in instruments],
                'Th price':[model.price_swaption(x) for x in instruments]})
print('Swaptions')
print(df)
# #Caps
# df=pd.DataFrame({'Item':instruments,
#                 'Mkt price':[x.mkt_price for x in instruments],
#                 'Th price':[model.price_cap(x) for x in instruments]})
# print('Caps')
# print(df)
