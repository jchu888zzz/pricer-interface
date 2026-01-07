import numpy as np
import QuantLib as ql

from Pricing.Rates.Payoffs import Funding
from Pricing.Rates.Payoffs.Types import Base,MinMax,Digit,FixedRate,RangeAccrual
from Pricing.Curves.Classic import Risky_Curve

def precomputation(calc_date:ql.Date,
                   contract:MinMax.MinMax|Digit.Digit|FixedRate.FixedRate|RangeAccrual.RangeAccrual,
                   model,risky_curve:Risky_Curve,structure_choice:str="Bond"):

    Base.prep_contract_common(calc_date, contract, risky_curve)
    
    Nbsimu=10000
    calendar=ql.Actual360()
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                        cal=calendar,Nbsimu=Nbsimu,seed=0)
    if contract.hasunderlying :
        undl_prep = Base.prep_undl(contract, model, data_rates, include_rates=True)
        contract._simu=undl_prep
        contract.fwds=np.mean(undl_prep['undl'],axis=1)
    else:
        contract._simu={'nbsimu':Nbsimu}

    contract.proba_recall=np.zeros_like(contract.pay_dates)
    contract.proba_recall[-1]=1
    contract.duration=contract.paygrid[-1]
    
    if structure_choice=='Bond':
        risky=True
        contract.zc=risky_curve.discount_factor(contract.pay_dates,risky)
        contract.funding_spread=Base.get_funding_spread(risky_curve,
                                                contract.pay_dates[-1],
                                                contract.funding_adjustment)
        
        return 
    
    if structure_choice=='Swap':
        if not hasattr(contract,"funding_spread"):
            raise ValueError(" funding_spread not found in input data")
        risky=False
        contract.zc=risky_curve.discount_factor(contract.pay_dates,risky)
        funding_leg=Funding.Leg(contract,contract.currency)
        funding_leg.precomputation(calc_date,model,data_rates)
        funding_leg.zc=risky_curve.discount_factor(funding_leg.pay_dates,risky)
        contract._funding_leg=funding_leg

        return 

    raise ValueError(f"{structure_choice} not implemented")

def compute_bond_price(contract,dic_arg:dict=None) -> float:
        """ Precomputation must be done before"""
        dic_arg = Base._validate_and_set_dic_arg(contract,dic_arg)
        cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"classic")
        contract.res_coupon = np.mean(cashflows, axis=0)
        contract.res_capital=contract.proba_recall
        prices = contract.res_coupon + contract.res_capital
        res = sum(prices * contract.zc)
        return res

def compute_swap_price(contract,dic_arg:dict=None) -> float:
        """ Precomputation must be done before and with the swap option"""
        if not hasattr(contract,"_funding_leg"):
            raise ValueError(f" contract must be prepare as a swap")

        dic_arg = Base._validate_and_set_dic_arg(contract,dic_arg)
        cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"classic")
        contract.res_coupon = np.mean(cashflows, axis=0)

        funding_leg = contract._funding_leg
        funding_leg.compute_values(contract.funding_spread)

        structure_price = sum(contract.res_coupon * contract.zc)
        funding_price = sum(funding_leg.coupons * funding_leg.zc)
        res = structure_price - funding_price

        return res


def solve_coupon(dic_prep:dict,swap:bool) -> tuple[float,float]:
    """
    Solve for optimal coupon based on swap price. Works for future contracts
    Returns (coupon, funding_spread).
    """
    contract=dic_prep['contract']
    risky_curve=dic_prep['risky_curve']
    target = contract.UF + contract.yearly_buffer * contract.paygrid[-1]

    res_spread = Base.get_funding_spread(risky_curve, contract.pay_dates[-1], contract.funding_adjustment)

    if swap:
        if "funding_leg" not in dic_prep.keys():
            raise ValueError("price_swap=True requires 'funding_leg' in dic_prep")
        
        funding_leg = dic_prep['funding_leg']
        funding_leg.compute_values(res_spread)
        funding_price = sum(funding_leg.coupons * funding_leg.zc)

        def func_to_solve(x: float):
            dic_arg = contract.update_arg_pricing(x, dic_prep['dic_arg'])
            cashflows = contract.compute_cashflows(dic_arg)
            res_coupon = np.mean(cashflows, axis=0)
            structure_price = sum(res_coupon *contract.zc)
            return (structure_price - funding_price + target) ** 2

        res_coupon = Base.optimize_coupon(func_to_solve)

    else:
        def func_to_solve(x: float):
            dic_arg = contract.update_arg_pricing(x, dic_prep['dic_arg'])
            cashflows = contract.compute_cashflows(dic_arg)
            coupons = np.mean(cashflows, axis=0)
            return (sum((coupons + contract.res_capital) * contract.zc) - (1 - target)) ** 2

        res_coupon = Base.optimize_coupon(func_to_solve)

    return res_coupon, res_spread