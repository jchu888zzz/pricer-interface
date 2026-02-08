import numpy as np
import QuantLib as ql

from Pricing.Utilities import InputConverter,Functions
from Pricing.Curves.Classic import Risky_Curve
from Pricing.Rates.Payoffs.Types import  Base
from Pricing.Rates.Payoffs import Funding,AutocallableFeature

class Tarn(Base.Payoff):

    def __init__(self,parameters:dict):
        self.get_common_parameters(parameters)
        self.coupon_lvl=InputConverter.set_param(parameters['coupon_level'],0)
        self.memory=False
        
        self.get_guaranteed_coupon_info(parameters)
        self.target=InputConverter.set_param(parameters['target'],0)
        self.call_dates=self.fix_dates
    
    def _set_dic_arg(self):
        return  {'coupon':self.coupon}
    
    def _compute_cashflows_and_stop_idxs(self,dic_arg:dict) -> tuple[np.ndarray,list]:
        coupon=dic_arg['coupon']
        undl=self._simu['undl']
        
        guaranteed_cashflows=np.zeros_like(undl.T)
        if hasattr(self,"guar_coupon"):
            guaranteed_cashflows[:,:self.NC]=self.guar_coupon
        
        cdt_cashflows=Base.compute_cdt_digit(undl,self.coupon_lvl,
                            self.infine,self.memory)*coupon
        cdt_cashflows[:,:self.NC]=False
        cashflows=guaranteed_cashflows+cdt_cashflows

        autocall_cdt=(np.cumsum(cashflows,axis=1) >=self.target )
        stop_idxs=Functions.first_occ_vec(autocall_cdt, True)
        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,self.infine)

        return cashflows,stop_idxs
    

    


    
