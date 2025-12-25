import numpy as np
import QuantLib as ql

import Pricing.Rates.Payoffs.Base as Base
import Pricing.Utilities.InputConverter as InputConverter
from Pricing.Rates import ResultHelper

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):
    contract=RangeAccrual(data)
    return Base.prep_callable_contract(calc_date,contract,model,risky_curve,risky)

def compute_bond_price(dic_prep:dict,risky_curve,risky:bool):
    return ResultHelper.compute_bond_price_callable(dic_prep,risky_curve,risky)

def compute_swap_price(dic_prep:dict,risky_curve):
    return ResultHelper.compute_swap_price_callable(dic_prep,risky_curve)

def solve_coupon_for_bond(dic_prep:dict,risky_curve,risky:bool):
    return ResultHelper.solve_coupon_for_bond_callable(dic_prep,risky_curve,risky)

def solve_coupon_for_swap(dic_prep:dict,risky_curve):
    return ResultHelper.solve_coupon_for_swap_callable(dic_prep,risky_curve)

class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                         param_contract,prep_model['risky_curve'],risky=True)
        if param_contract['structure_type']=='Bond':
            return compute_bond_price(dic_prep,prep_model['risky_curve'],risky=True)
        else:
            return compute_swap_price(dic_prep,prep_model['risky_curve'])
        
    def solve_coupon(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                         param_contract,prep_model['risky_curve'],risky=True)
        
        def update_dic_prep(coupon,spread) ->dict:
            dic_prep_new=dic_prep.copy()
            dic_prep_new['contract'].coupon=coupon
            dic_prep_new['contract'].funding_spread=spread
            return dic_prep_new

        if param_contract['structure_type']=='Bond':
            coupon,spread=solve_coupon_for_bond(dic_prep,prep_model['risky_curve'],risky=True)
            dic_prep_new=update_dic_prep(coupon,spread)
            return compute_bond_price(dic_prep_new,prep_model['risky_curve'],risky=True)
        else:
            coupon,spread=solve_coupon_for_swap(dic_prep,prep_model['risky_curve'])
            dic_prep_new=update_dic_prep(coupon,spread)
            return compute_swap_price(dic_prep_new,prep_model['risky_curve'])

class RangeAccrual(Base.Payoff) :

    def __init__(self,parameters:dict[str:str]):
        self.typename='Range Accrual'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)
        self.lower_bound=InputConverter.set_param(parameters['lower_bound'],0)
        self.upper_bound=InputConverter.set_param(parameters['upper_bound'],0)
        self.fixing_depth=52 #weekly
    
    def compute_densities(self,undl:np.ndarray):
        res=np.zeros((undl.shape[0],undl.shape[2]))
        def InBound(values):
            res=0 
            for x in values:
                if self.lower_bound<=x<=self.upper_bound :
                    res+=1
            return res
    
        for i in range(undl.shape[0]):
            res[i]=np.array([ InBound(undl[i][:,j]) for j in range(undl.shape[2]) ])/undl.shape[1]
        
        return res 
    
    def compute_cashflows(self,dic_arg:dict) -> np.ndarray:    
        coupon=dic_arg['x']
        densities=dic_arg['densities']
        return coupon*densities.T*np.tile(self.delta,(densities.shape[1],1))
    
    def update_arg_pricing(self,coupon:float,dic_arg:dict) -> dict:
        res=dic_arg.copy()
        res.update({'x':coupon})
        return res



