import sys
from pathlib import Path

pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import QuantLib as ql
import pandas as pd

from Pricing.Utilities import Display
from Pricing.Rates import GetResults
from Pricing.Rates.Model import  HullWhite
from Pricing.Rates.Payoffs import AutocallableFeature
from Pricing.Rates.Payoffs.Types import Base,Autocall

# Configuration
# DATA_PATH=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
DATA_PATH = r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
CALC_DATE = ql.Date(11, 11, 2025)

input={
"_source_page": "Rate",
"_source_tab": "Autocall",
"param": {
"NC": "1",
"UF": "2.0%",
"autocall_level": "2.5%",
"coupon_level": "3.5%",
"currency": "EUR",
"fixing_days_offset": "-5",
"fixing_type": "in arrears",
"frequency": "Annually",
"in-fine": "false",
"issue_date": "07.01.2026",
"maturity": "10",
"memory_effect": "false",
"solving_choice": "Solve coupon",
"underlying1": "EUR CMS 5Y",
"yearly_buffer": "0.0%"
}
}

def get_model(mkt_data, params):
    underlying = params.get('underlying1')
    return HullWhite.get_model(mkt_data['calc_date'], mkt_data, params['currency'], underlying)

def solve_autocallable(contract, prep_model, structure_choice):
    risky_curve = prep_model['risky_curve']

    AutocallableFeature.precomputation(
        prep_model['calc_date'], contract, prep_model['model'],
        risky_curve, structure_choice=structure_choice)

    if structure_choice == "Bond":
        coupon,funding_spread=Autocall.solve_coupon_bond(contract, risky_curve)
        contract.coupon=coupon,
        contract.funding_spread=funding_spread
        price = AutocallableFeature.compute_bond_price(contract)
    else:  # Swap
        coupon,funding_spread=Autocall.solve_coupon_swap(contract, risky_curve)
        contract.coupon=coupon,
        contract.funding_spread=funding_spread
        fund_res = Base.organize_funding_table(contract._funding_leg)
        print(pd.DataFrame(fund_res))
        price = AutocallableFeature.compute_swap_price(contract)

    return price

def check_price(structure_choice="Swap"):
    mkt_data = GetResults.retrieve_data(path_folder=DATA_PATH, date=CALC_DATE)
    params = input['param']

    prep_model = get_model(mkt_data, params)
    contract = Autocall.Autocall(params)

    price=solve_autocallable(contract,prep_model,structure_choice)
    detailed_result = {'price': price}
    detailed_result.update(Base.organize_contract_result(contract))
    print("coupon",contract.coupon)
    Display.display_pricing_results(detailed_result)

    return detailed_result

if __name__ == "__main__":
    check_price("Swap")