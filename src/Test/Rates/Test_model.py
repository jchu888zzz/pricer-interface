import sys
from pathlib import Path
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd

from Pricing.Rates import GetResults
from Pricing.Curves import Classic
from Pricing.Rates.Model import  HullWhite
from Pricing.Rates import Instruments


# Configuration
# DATA_PATH=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
DATA_PATH = r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
CALC_DATE = ql.Date(11, 11, 2025)

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

def get_instruments(mkt_data,curve,input_param):

    if  "underlying1" in input_param.keys():
        _,rate_type,tenor=input_param["underlying1"].split()
        if rate_type=='CMS':
            instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                            curve,CALC_DATE,input_param['currency'])
            instruments=[x for x in instruments if x.strike_type=='ATM' and x.tenor==tenor]
        elif rate_type=='Euribor':
            instruments=Instruments.select_and_prepare_caps(mkt_data['caps'],curve,CALC_DATE,input_param['currency'])
        else:
            raise ValueError(f"not implemented")
    else:
        instruments=Instruments.select_and_prepare_swaptions(mkt_data['swaption'],
                                            curve,CALC_DATE,input_param['currency'])
        instruments=[x for x in instruments if x.strike_type=='ATM']
    return instruments

def test_calibration():
    """Main pricing workflow."""
    mkt_data = GetResults.retrieve_data(path_folder=DATA_PATH, date=CALC_DATE)
    input_param=params = input['param']
    curve,risky_curve=Classic.get_curves(CALC_DATE,mkt_data,input_param['currency'],'Classical')
    instruments=get_instruments(mkt_data,curve,input_param)

    model=HullWhite.Calibration(curve,instruments)
    df=pd.DataFrame({'Item':instruments,
                'Mkt price':[x.mkt_price for x in instruments],
                'Th price':[model.price_swaption(x)  if isinstance(x,Instruments.Swaption) else model.price_cap(x) 
                             for x in instruments]})
    return df

if __name__ == "__main__":
    df=test_calibration()
    print(df)