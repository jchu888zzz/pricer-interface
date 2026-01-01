import numpy as np
import QuantLib as ql
from sklearn.neighbors import KNeighborsRegressor

from Pricing.Utilities import InputConverter
from Pricing.Rates.Payoffs import Base
from . import CallableFeature

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):
    contract=Digit(data)
    return CallableFeature.prep_callable_contract(calc_date,contract,model,risky_curve,risky)


REGRESSOR_CLASS=KNeighborsRegressor(n_neighbors=20)
def compute_price(dic_prep:dict,risky_curve):
    return CallableFeature.compute_price(dic_prep,risky_curve,
                                            basis_option='polynomial',
                                            regressor_class=REGRESSOR_CLASS)

def solve_coupon(dic_prep:dict,risky_curve):
    return CallableFeature.solve_coupon(dic_prep,risky_curve,
                                        basis_option='polynomial',
                                        regressor_class=REGRESSOR_CLASS)


class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                                param_contract,prep_model['risky_curve'],risky=True)

        return compute_price(dic_prep,prep_model['risky_curve'])
        
    def solve_coupon(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                                param_contract,prep_model['risky_curve'],risky=True)
        
        def update_dic_prep(coupon,spread) ->dict:
            dic_prep_new=dic_prep.copy()
            dic_prep_new['contract'].coupon=coupon
            dic_prep_new['contract'].funding_spread=spread
            return dic_prep_new

        coupon,spread=solve_coupon(dic_prep,prep_model['risky_curve'])
        dic_prep_new=update_dic_prep(coupon,spread)
        return compute_price(dic_prep_new,prep_model['risky_curve'])

class Digit(Base.Payoff):
    def __init__(self,parameters:dict):
        self.get_common_parameters(parameters)
        self.coupon_lvl=InputConverter.set_param(parameters['coupon_level'],0)
        self.get_memory_effect(parameters)
        self.get_callable_info(parameters)
    
    def compute_cashflows(self,dic_arg:dict) -> np.ndarray:
        coupon=dic_arg['x']
        undl=dic_arg['undl']
        cdt=Base.compute_cdt_digit(undl,self.coupon_lvl,
                                    self.infine,self.memory)
        return coupon*cdt
    
    def update_arg_pricing(self,coupon:float,dic_arg:dict) -> dict:
        res=dic_arg.copy()
        res.update({'x':coupon})
        return res
