import numpy as np
import QuantLib as ql

from Pricing.Rates.Payoffs import Base
from Pricing.Utilities import InputConverter
from Pricing.Rates.Payoffs.Callable import CallableFeature
from Pricing.Rates import ResultHelper

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):
    contract=MinMax(data)
    return Base.prep_callable_contract(calc_date,contract,model,risky_curve,risky)

def compute_bond_price(dic_prep:dict,risky_curve,risky:bool):
    
    contract=dic_prep['contract']
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky)

    cashflows=contract.compute_cashflows(dic_prep['dic_arg'])

    if 'dic_arg_helper' in dic_prep.keys():
        deg=5
        regressions=CallableFeature.get_regression_for_bond_with_undl(contract,
                                                                    dic_prep['dic_arg_helper'],deg)
        
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_prep['dic_arg'],
                                                                deg,include_principal=True)
        
        contract.compute_cashflows(dic_prep['dic_arg'])
        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        contract.res_coupon=np.mean(cashflows,axis=0)
        contract.proba_recall=contract.compute_recall_proba(stop_idxs)
        contract.res_capital=Base.compute_bond_measure_change(dic_prep['dic_arg']['measure_change_factor'],
                                                                stop_idxs)

    contract.res_coupon=np.mean(cashflows,axis=0)
    prices=contract.res_coupon+contract.res_capital
    price=sum(prices*ZC)
    
    res=ResultHelper.organize_structure_table(contract,ZC)
    res['Price']=price
    res["duration"]=sum(contract.proba_recall*contract.paygrid)
    res["funding_spread"]=ResultHelper.get_funding_spread_early_redemption(risky_curve,
                                                                contract.paygrid,contract.proba_recall)
    return res

def compute_swap_price(dic_prep:dict,risky_curve):
    
    contract=dic_prep['contract']
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)

    cashflows=contract.compute_cashflows(dic_prep['dic_arg'])

    funding_leg=dic_prep['funding_leg']
    funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    if not 'dic_arg_helper' in dic_prep.keys():
        funding_leg.compute_values(contract.funding_spread)
        
    if 'dic_arg_helper' in dic_prep.keys():
        deg=5
        regressions=CallableFeature.get_regression_for_bond_with_undl(contract,
                                                                    dic_prep['dic_arg_helper'],deg)
        
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_prep['dic_arg'],
                                                                deg,include_principal=True)
        
        contract.compute_cashflows(dic_prep['dic_arg'])
        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        contract.proba_recall=contract.compute_recall_proba(stop_idxs)
        
        funding_leg.compute_values_for_early_redemption(stop_idxs,contract.funding_spread)

    contract.res_coupon=np.mean(cashflows,axis=0)
    structure_price=sum(contract.res_coupon*ZC)
    funding_price=sum(funding_leg.coupons*funding_ZC)
    price=structure_price-funding_price
    
    return {'Structure':ResultHelper.organize_structure_table(contract,ZC),
            'Funding':ResultHelper.organize_funding_table(funding_leg,funding_ZC),
            'Price':price,
            'duration':sum(contract.proba_recall*contract.paygrid),
            'funding_spread':contract.funding_spread}

class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                            param_contract,prep_model['risky_curve'],risky=True)
        if param_contract['structure_type']=='Bond':
            return compute_bond_price(dic_prep,prep_model['risky_curve'],risky=True)
        else:
            return compute_swap_price(dic_prep,prep_model['risky_curve'])


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
