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
input=test_range_swap

input,res=GetResults.compute_result_rate(mkt_data,input)
Display.display_pricing_results(res)