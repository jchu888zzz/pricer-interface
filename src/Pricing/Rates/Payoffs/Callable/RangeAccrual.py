import numpy as np
import QuantLib as ql
from sklearn.neighbors import KNeighborsRegressor

import Pricing.Rates.Payoffs.Base as Base
import Pricing.Utilities.InputConverter as InputConverter
from . import CallableFeature

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):
    contract=RangeAccrual(data)
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



