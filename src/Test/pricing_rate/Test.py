import QuantLib as ql
import pandas as pd
import sys

from pathlib import Path

pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

from Pricing.Rates import GetResults
from Pricing.Utilities import Display

from contracts import *

DataPath =r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
calc_date=ql.Date(11,11,2025)
mkt_data=GetResults.retrieve_data(path_folder=DataPath,date=calc_date)
#select test from contracts
#♥input=test_tarn_bond

input={
 "_source_page": "Rate",
 "_source_tab": "Tarn",
 "param": {
  "coupon": "5.0%",
  "coupon_level": "3.0%",
  "currency": "EUR",
  "fixing_days_offset": "-5",
  "fixing_type": "in arrears",
  "frequency": "Annually",
  "guaranteed_coupon": "0.0%",
  "in-fine": "false",
  "issue_date": "27.12.2025",
  "maturity": "10",
  "nb_guaranteed_coupon": "0",
  "solving_choice": "Price",
  "structure_type": "Bond",
  "target": "10.0%",
  "underlying1": "EUR CMS 5Y"
 }
}


input,res=GetResults.compute_result_rate(mkt_data,input)
Display.display_pricing_results(res)