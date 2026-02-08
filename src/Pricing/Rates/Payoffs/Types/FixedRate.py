import numpy as np
import QuantLib as ql
from sklearn.linear_model import Ridge

from Pricing.Utilities import Functions
from Pricing.Rates.Payoffs.Types import Base

def compute_swaption_price(DF:float,K:float,x:float,Pt_T:np.ndarray,delta:np.ndarray,side:str="sell"):
    if side=="buy":
        return DF*np.maximum(x-K,0)*np.sum(delta*Pt_T[:,1:],axis=1)
    if side=="sell":
        return DF*np.maximum(K-x,0)*np.sum(delta*Pt_T[:,1:],axis=1)
    else: 
        raise ValueError(' Invalid input {side}')

class FixedRate(Base.Payoff) :

    def __init__(self,parameters:dict):
        self.fixing_type='in arrears'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)

        self._regressor_class=Ridge(alpha=1.0,fit_intercept=True)

    def _set_dic_arg(self):
        return  {'coupon':self.coupon}   

    def _compute_cashflows(self,dic_arg:dict) -> dict:
        """ Precomputation must be done before"""

        coupon=dic_arg['coupon']
        cf=coupon*self.delta

        def func_cf(nbsimu)->np.ndarray:
            if self.infine:
                return np.tile(np.cumsum(cf),(nbsimu,1))
            else:
                return np.tile(cf,(nbsimu,1))
        
        res=dict()
        simu=self._simu
        res["classic"]=func_cf(simu['nbsimu'])
        if not hasattr(self,"call_dates"):
            return res
        
        #Update swaption based on coupon value
        simu_helper=self._simu_helper
        res["helper"]=func_cf(simu_helper['nbsimu'])

        undl_helper=[None]*len(self.pay_dates)
        undl=[None]*len(self.pay_dates)
        for i,d in enumerate(self.call_dates):
            idx=Functions.find_idx(self.fix_dates,d)
            undl[idx]=compute_swaption_price(simu["DF"][i],coupon,simu["swap"][i],
                                    simu["Pt_T"][i],simu["delta"][i])
            
            undl_helper[idx]=compute_swaption_price(simu_helper["DF"][i],coupon,simu_helper["swap"][i],
                                    simu_helper["Pt_T"][i],simu_helper["delta"][i])

        simu["undl"]=undl
        simu_helper["undl"]=undl_helper

        return res
    