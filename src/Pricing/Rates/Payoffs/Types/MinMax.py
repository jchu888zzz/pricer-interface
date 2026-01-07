import numpy as np
import QuantLib as ql
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from Pricing.Rates.Payoffs.Types import Base
from Pricing.Utilities import InputConverter

class MinMax(Base.Payoff) :
    def __init__(self,parameters:dict[str:str]):
        self.typename='Min Max'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)
        if 'floor' in parameters.keys():
            self.floor=InputConverter.set_param(parameters['floor'],0)
        if 'cap' in parameters.keys():
            self.cap=InputConverter.set_param(parameters['cap'],0)

        self._regressor_class=KNeighborsRegressor(n_neighbors=30)
    
    def _set_dic_arg(self):
        return  {'floor':self.floor,'cap':self.cap} 

    def _compute_cashflows(self,dic_arg:dict) -> dict[str:np.ndarray]:
        """ Precomputation must be done before"""
        floor=dic_arg['floor']
        cap=dic_arg['cap']

        undl=self._simu['undl']
        res=dict()
        #broadcast delta to fit cashflows
        func=lambda x: np.maximum(floor,np.minimum(x.T,cap))*np.tile(self.delta,(x.shape[1],1)) 
        res["classic"]=func(undl)
        if not hasattr(self,"call_dates"):
            return res
        
        undl_helper=self._simu_helper['undl']
        res["helper"]=func(undl_helper)
        return res
    