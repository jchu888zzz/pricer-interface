import numpy as np
import QuantLib as ql
from sklearn.linear_model import Ridge

from Pricing.Rates.Payoffs import Base
from Pricing.Utilities import InputConverter
from . import CallableFeature1,CallableFeature

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):
    contract=MinMax(data)
    return CallableFeature.prep_callable_contract(calc_date,contract,model,risky_curve,risky)
    #return CallableFeature1.prep_callable_contract(calc_date,contract,model,risky_curve,risky)

REGRESSOR_CLASS=Ridge(alpha=0.8,fit_intercept=True)
def compute_price(dic_prep:dict,risky_curve,basis_option:str='polynomial'):
    """
    Compute price for callable MinMax bond or swap.
    For bonds: risky=True determines discount factor
    For swaps: risky=False, includes funding leg pricing
    """
    
    contract=dic_prep['contract']
    # Check if this is a swap (has funding_leg) or bond
    is_swap='funding_leg' in dic_prep.keys()
    
    if is_swap:
        ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)
        funding_leg=dic_prep['funding_leg']
        funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    else:
        ZC=risky_curve.Discount_Factor(contract.paygrid,risky=True)

    cashflows=contract.compute_cashflows(dic_prep['dic_arg'])

    if 'dic_arg_helper' in dic_prep.keys():
        deg=3
        regressions=CallableFeature.get_regression_for_bond_with_undl(contract,
                                                                    dic_prep['dic_arg_helper'],deg,
                                                                    basis_option=basis_option,
                                                                    regressor_class=REGRESSOR_CLASS)
        
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_prep['dic_arg'],
                                                                deg,include_principal=True,
                                                                basis_option=basis_option)
        
        contract.compute_cashflows(dic_prep['dic_arg'])
        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        contract.res_coupon=np.mean(cashflows,axis=0)
        contract.proba_recall=contract.compute_recall_proba(stop_idxs)
        
        if not is_swap:
            contract.res_capital=Base.compute_bond_measure_change(dic_prep['dic_arg']['measure_change_factor'],
                                                                    stop_idxs)
        else:
            funding_leg.compute_values_for_early_redemption(stop_idxs,contract.funding_spread)
    else:
        if is_swap:
            funding_leg.compute_values(contract.funding_spread)

    contract.res_coupon=np.mean(cashflows,axis=0)
    if is_swap:
        structure_price=sum(contract.res_coupon*ZC)
        funding_price=sum(funding_leg.coupons*funding_ZC)
        price=structure_price-funding_price
    else:
        prices=contract.res_coupon+contract.res_capital
        price=sum(prices*ZC)
        
    res=dict()
    res['table']=Base.organize_structure_table(contract,ZC)
    res['price']=price
    res["duration"]=sum(contract.proba_recall*contract.paygrid)
    res["funding_spread"]=Base.get_funding_spread_early_redemption(risky_curve,
                                                                contract.paygrid,contract.proba_recall,
                                                                contract.funding_adjustment)
    
    if is_swap:
        res['funding_table']=Base.organize_funding_table(funding_leg,funding_ZC)
    
    return res


class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                            param_contract,prep_model['risky_curve'],risky=True)

        return compute_price(dic_prep,prep_model['risky_curve'])


class MinMax(Base.Payoff) :

    def __init__(self,parameters:dict[str:str]):
        self.typename='Min Max'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)
        if 'floor' in parameters.keys():
            self.floor=InputConverter.set_param(parameters['floor'],0)
        if 'cap' in parameters.keys():
            self.cap=InputConverter.set_param(parameters['cap'],0)
    
    def compute_cashflows(self,dic_arg:dict) -> np.ndarray:
        undl=dic_arg['undl']
        return np.maximum(self.floor,np.minimum(undl.T,self.cap))*np.tile(self.delta,(undl.shape[1],1))
