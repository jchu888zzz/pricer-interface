import numpy as np

from Pricing.Utilities import InputConverter,Functions
from Pricing.Curves.Classic import Risky_Curve
from Pricing.Rates.Payoffs.Types import  Base

def _precomputation_solve_coupon(contract:Autocall,risky_curve:Risky_Curve) ->tuple[float]:
    target = contract.UF + contract.yearly_buffer * contract.paygrid[-1]
    # Pre-compute stop indices (independent of coupon)
    undl = contract._simu["undl"]
    stop_idxs = contract.compute_stop_idxs(undl)
    contract.proba_recall = contract.compute_recall_proba(stop_idxs)

    contract.res_capital=Base.compute_bond_measure_change(contract._measure_change_factor,stop_idxs)
    # Pre-compute funding spread (independent of coupon)
    res_funding = Base.get_funding_spread_early_redemption(
        risky_curve, contract.pay_dates,
        contract.proba_recall, contract.funding_adjustment
    )
    return target,stop_idxs,res_funding

def solve_coupon_bond(contract:Autocall, risky_curve:Risky_Curve) -> tuple[float]:
    """ Precomputation must be done before"""
    
    target,stop_idxs,res_funding=_precomputation_solve_coupon(contract,risky_curve)
    price_target=1-target
    # Pre-compute capital value (constant across iterations)
    capital_zc = contract.res_capital * contract.zc
    capital_value = np.sum(capital_zc)

    def func_to_solve(x: float) -> float:
        cashflows, _ = contract._compute_cashflows_and_stop_idxs({'coupon': x}, stop_idxs)
        coupons_value = np.sum(np.mean(cashflows, axis=0) * contract.zc)
        return (coupons_value  + capital_value - price_target) ** 2

    res_coupon = Base.optimize_coupon(func_to_solve)
    return res_coupon, res_funding

def solve_coupon_swap(contract:Autocall,risky_curve:Risky_Curve) -> tuple[float, float]:
    """ Precomputation must be done before"""
    target,stop_idxs,res_funding=_precomputation_solve_coupon(contract,risky_curve)

    # Pre-compute funding leg value (constant across iterations)
    funding_leg=contract._funding_leg
    funding_leg.compute_values_for_early_redemption(stop_idxs,res_funding)
    funding_price=sum(funding_leg.coupons*funding_leg.zc)

    # Pre-compute measure change factor for structure leg (matches funding leg treatment)
    mcf = contract._measure_change_factor  # shape: (num_dates, num_sims)
    zc = contract.zc

    def func_to_solve(x:float):
        cashflows, _ = contract._compute_cashflows_and_stop_idxs({'coupon': x}, stop_idxs)
        # Apply measure change: average of (cashflow / mcf) per date, then discount
        adjusted_coupons = np.mean(cashflows*mcf.T, axis=0)
        structure_price = np.sum(adjusted_coupons * zc)
        return (structure_price - funding_price + target)**2

    res_coupon = Base.optimize_coupon(func_to_solve)
    return res_coupon, res_funding

class Autocall(Base.Payoff):

    def __init__(self,parameters:dict[str:str]):
        self.get_common_parameters(parameters)
        self.coupon_lvl=InputConverter.set_param(parameters['coupon_level'],0)
        self.autocall_lvl=InputConverter.set_param(parameters['autocall_level'],0)
        self.get_memory_effect(parameters)
        if 'call_dates' in parameters.keys():
            self.call_dates=parameters['call_dates']
            return
        if 'NC' in parameters.keys():
            non_call=max(int(parameters['NC'])-1,0)
            self.call_dates=self.fix_dates[non_call:-1]

    def _set_dic_arg(self):
        return  {'coupon':self.coupon}

    def compute_stop_idxs(self,undl:np.ndarray) -> list:
        """ undl shape (nb time steps, nb simu)"""
        autocall_cdt=(undl.T <=self.autocall_lvl)
        non_call=len(self.fix_dates) -(len(self.call_dates)+1)
        autocall_cdt[:,:non_call]=0 # set period before call to 0
        return Functions.first_occ_vec(autocall_cdt, True) 
    
    def _compute_cashflows_and_stop_idxs(self,dic_arg:dict,stop_idxs:list=None) -> tuple[np.ndarray,list]:
        """ dic_arg must contain necessary arguments for pricing cashflows """
        coupon=dic_arg['coupon']
        undl=self._simu['undl']
        
        if not isinstance(stop_idxs,list):
            stop_idxs=self.compute_stop_idxs(undl)
        
        cashflow_cdt=Base.compute_cdt_digit(undl,self.coupon_lvl,self.infine,self.memory)
        cashflow_cdt=Base.adjust_to_stop_idxs(cashflow_cdt,stop_idxs,self.infine)
        cashflows=coupon*cashflow_cdt

        return cashflows,stop_idxs
    