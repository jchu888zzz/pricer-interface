import numpy as np
import QuantLib as ql
from sklearn.neighbors import KNeighborsRegressor

from Pricing.Rates.Payoffs.Types import Base
import Pricing.Utilities.InputConverter as InputConverter

class RangeAccrual(Base.Payoff) :

    def __init__(self,parameters:dict[str:str]):
        self.typename='Range Accrual'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)
        self.lower_bound=InputConverter.set_param(parameters['lower_bound'],0)
        self.upper_bound=InputConverter.set_param(parameters['upper_bound'],0)
        self.fixing_depth=52 #weekly

        self._regressor_class=KNeighborsRegressor(n_neighbors=20)

    def _set_dic_arg(self):
        return  {'coupon':self.coupon}  

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
    
    def _compute_cashflows(self,dic_arg:dict) -> np.ndarray:    
        coupon=dic_arg['coupon']

        #broadcast delta to fit cashflows
        func=lambda x: coupon*x.T*np.tile(self.delta,(x.shape[1],1))
        res=dict()
        densities=self._simu['densities']
        res["classic"]=func(densities)
        if not hasattr(self,"call_dates"):
            return res
        
        densities_helper=self._simu_helper['densities']
        res["helper"]=func(densities_helper)
        return res
    