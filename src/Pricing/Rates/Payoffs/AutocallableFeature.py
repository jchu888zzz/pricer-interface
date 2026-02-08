import numpy as np
import QuantLib as ql

from Pricing.Utilities import InputConverter,Functions
from Pricing.Rates.Payoffs.Types import  Base
from Pricing.Curves.Classic import Risky_Curve
from Pricing.Rates.Payoffs import Funding

def precomputation(calc_date:ql.Date,
                   contract,
                   model,risky_curve:Risky_Curve,structure_choice:str="Bond"):
    """ Common Precomputation for autocallable contracts """

    Base.prep_contract_common(calc_date, contract,risky_curve)

    calendar=ql.Actual360()
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                        cal=calendar,Nbsimu=10000,seed=0)
    
    undl_prep = Base.prep_undl(contract, model, data_rates, include_rates=True)
    contract._simu=undl_prep
    contract.fwds=np.mean(undl_prep['undl'],axis=1)

    measure_change_factor=np.array([Base.compute_measure_change_factor(model,undl_prep['rates'][i],t,contract.paygrid[-1])
                                    for i,t in enumerate(contract.paygrid) ])[:,:,0]
    contract._measure_change_factor=measure_change_factor

    if structure_choice=='Bond':
        risky=True
        contract.zc=risky_curve.discount_factor(contract.pay_dates,risky)        
        return 
    
    if structure_choice=='Swap':
        risky=False
        contract.zc=risky_curve.discount_factor(contract.pay_dates,risky)
        funding_leg=Funding.Leg(contract,contract.currency)
        funding_leg.precomputation(calc_date,model,data_rates)
        funding_leg.zc=risky_curve.discount_factor(funding_leg.pay_dates,risky)
        contract._funding_leg=funding_leg

        return 

    raise ValueError(f"{contract.structure_type} not implemented")

def compute_bond_price(contract,dic_arg:dict=None) -> float:
    """ Precomputation must be done before"""
    dic_arg = Base._validate_and_set_dic_arg(contract,dic_arg)
    cashflows,stop_idxs=contract._compute_cashflows_and_stop_idxs(dic_arg)
    contract.proba_recall=contract.compute_recall_proba(stop_idxs)
    
    contract.res_coupon=np.mean(cashflows,axis=0)
    contract.res_capital=Base.compute_bond_measure_change(contract._measure_change_factor,stop_idxs)
    res=np.sum((contract.res_coupon+contract.res_capital)*contract.zc)

    return res

def compute_swap_price(contract,dic_arg:dict=None) -> float:
    """ Precomputation must be done before"""
    dic_arg = Base._validate_and_set_dic_arg(contract,dic_arg)
    cashflows,stop_idxs=contract._compute_cashflows_and_stop_idxs(dic_arg)
    contract.proba_recall=contract.compute_recall_proba(stop_idxs)

    contract.res_coupon=np.mean(cashflows,axis=0)
    structure_price=sum(contract.res_coupon*contract.zc)
    funding_leg=contract._funding_leg
    funding_leg.compute_values_for_early_redemption(stop_idxs,contract.funding_spread)
    funding_price=sum(funding_leg.coupons*funding_leg.zc)

    res=structure_price-funding_price
    return res
