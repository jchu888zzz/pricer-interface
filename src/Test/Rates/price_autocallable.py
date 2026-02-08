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
from Pricing.Rates.Payoffs.Types import Base,Autocall,Tarn

# Configuration
# DATA_PATH=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\6_Lexifi\Market_data"
DATA_PATH = r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
CALC_DATE = ql.Date(11, 11, 2025)

PAYOFF_MAPPING={'Autocall':Autocall.Autocall,
                'Tarn':Tarn.Tarn}

input={
"_source_page": "Rate",
"_source_tab": "Tarn",
"param": {
"coupon": "5.0%",
"coupon_level": "2.5%",
"currency": "EUR",
"fixing_days_offset": "-5",
"fixing_type": "in arrears",
"frequency": "Annually",
"guaranteed_coupon": "4.0%",
"in-fine": "false",
"issue_date": "07.01.2026",
"maturity": "10",
"nb_guaranteed_coupon": "0",
"solving_choice": "Price",
"target": "10.0%",
"underlying1": "EUR CMS 5Y"
}
}
input["param"]["funding_spread"]="90bps"

PAYOFF_MAPPING={'Autocall':Autocall.Autocall,
                'Tarn':Tarn.Tarn}


def get_model(mkt_data, params):
    underlying = params.get('underlying1')
    return HullWhite.get_model(
        mkt_data['calc_date'], mkt_data, params['currency'], underlying
    )
def price_autocallable(contract, prep_model, structure_choice):
    risky_curve = prep_model['risky_curve']

    AutocallableFeature.precomputation(
        prep_model['calc_date'], contract, prep_model['model'],
        risky_curve, structure_choice=structure_choice)

    if structure_choice == "Bond":
        price = AutocallableFeature.compute_bond_price(contract)
        contract.funding_spread = Base.get_funding_spread_early_redemption(
            risky_curve, contract.pay_dates,
            contract.proba_recall, contract.funding_adjustment
        )
    else:  # Swap
        price = AutocallableFeature.compute_swap_price(contract)
        fund_res = Base.organize_funding_table(contract._funding_leg)
        print(pd.DataFrame(fund_res))

    return price

def run_pricing(structure_choice="Swap"):
    mkt_data = GetResults.retrieve_data(path_folder=DATA_PATH, date=CALC_DATE)
    params = input['param']

    prep_model = get_model(mkt_data, params)
    payoff_class = PAYOFF_MAPPING.get(input['_source_tab'])
    contract = payoff_class(params)

    price = price_autocallable(contract, prep_model, structure_choice)

    detailed_result = {'price': price}
    detailed_result.update(Base.organize_contract_result(contract))
    Display.display_pricing_results(detailed_result)

    return detailed_result

if __name__ == "__main__":
    run_pricing("Bond")
