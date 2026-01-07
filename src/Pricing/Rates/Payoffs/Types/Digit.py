import numpy as np
import QuantLib as ql
from sklearn.neighbors import KNeighborsRegressor

from Pricing.Utilities import InputConverter
from Pricing.Rates.Payoffs.Types import Base

class Digit(Base.Payoff):
    def __init__(self,parameters:dict):
        self.get_common_parameters(parameters)
        self.coupon_lvl=InputConverter.set_param(parameters['coupon_level'],0)
        self.get_memory_effect(parameters)
        self.get_callable_info(parameters)

        self._regressor_class=KNeighborsRegressor(n_neighbors=30)
    
    def _set_dic_arg(self):
        return  {'coupon':self.coupon}

    def _compute_cashflows(self,dic_arg:dict) -> np.ndarray:
        coupon=dic_arg['coupon']
        undl=self._simu['undl']

        res=dict()
        res["classic"]=coupon*Base.compute_cdt_digit(undl,self.coupon_lvl,self.infine,self.memory)
        if not hasattr(self,"call_dates"):
            return res
        
        undl_helper=self._simu_helper['undl']
        res["helper"]=coupon*Base.compute_cdt_digit(undl_helper,self.coupon_lvl,self.infine,self.memory)
        return res
    