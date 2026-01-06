import numpy as np
import QuantLib as ql
from sklearn.linear_model import Ridge

from Pricing.Utilities import Functions
from Pricing.Rates.Payoffs import Base
from Pricing.Rates import Funding
from . import CallableFeature

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):
    contract=FixedRate(data)
    contract._update(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    contract.compute_funding_adjustment(calc_date)
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                                    cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=0)
    
    res=dict()
    if contract.structure_type=='Swap':
        funding_leg=Funding.Leg(contract,contract.currency)
        funding_leg.precomputation(calc_date,model,data_rates)
        res.update({'funding_leg':funding_leg})
    
    if not hasattr(contract,'call_dates'):
        
        contract.proba_recall=np.zeros_like(contract.pay_dates)
        contract.proba_recall[-1]=1
        contract.res_capital=contract.proba_recall
        contract.duration=contract.paygrid[-1]
        contract.funding_spread=Base.get_funding_spread(risky_curve,
                                                        contract.pay_dates[-1],
                                                        contract.funding_adjustment)
        res.update({'contract':contract,
                    'dic_arg':{'nbsimu':data_rates['rates'].shape[0]}})
        return res

    dic_arg=model.compute_prep_for_swaption_from_rates(contract,data_rates,
                                            daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis),
                                            include_rates=True)
    CallableFeature.prep_discount_factor_from_rates(contract,model,risky_curve,dic_arg,risky)
    data_rates_helper=model.generate_rates(calc_date,contract.pay_dates[-1],cal=ql.Thirty360(ql.Thirty360.BondBasis),
                                Nbsimu=1000,seed=42)
    
    dic_arg_helper=model.compute_prep_for_swaption_from_rates(contract,data_rates_helper,
                                        daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis),
                                        include_rates=True)
    CallableFeature.prep_discount_factor_from_rates(contract,model,risky_curve,dic_arg_helper,risky)

    contract.paygrid=[risky_curve.calendar.yearFraction(risky_curve.calc_date,d) for d in contract.pay_dates]
    measure_change_factor=np.array([Base.compute_measure_change_factor(model,dic_arg['rates'][i],t,contract.paygrid[-1]) 
                                for i,t in enumerate(contract.paygrid) ])[:,:,0]
    dic_arg['measure_change_factor']=measure_change_factor

    res.update({'contract':contract,
            'dic_arg_helper':dic_arg_helper,
            'dic_arg':dic_arg})
    return res

REGRESSOR_CLASS=Ridge(alpha=0.5,fit_intercept=True)
def compute_price(dic_prep:dict,risky_curve):
    return CallableFeature.compute_price(dic_prep,risky_curve,
                                            basis_option='polynomial',
                                            regressor_class=REGRESSOR_CLASS)

def solve_coupon(dic_prep:dict,risky_curve):
    return CallableFeature.solve_coupon(dic_prep,risky_curve,
                                        basis_option='polynomial',
                                        regressor_class=REGRESSOR_CLASS)

class FixedRate(Base.Payoff) :

    def __init__(self,parameters:dict):
        self.fixing_type='in arrears'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)    

    def compute_cashflows(self,dic_arg:dict):
        coupon=dic_arg['x']
        nb_simu=dic_arg['nbsimu']
        res=coupon*self.delta

        if self.infine:
            return np.tile(np.cumsum(res),(nb_simu,1))
        else:
            return np.tile(res,(nb_simu,1))
        
    def update_arg_pricing(self,coupon:float,dic_arg:dict|None,side:str='sell') -> dict:
        
        if not isinstance(dic_arg,dict):
            res.update({'x':coupon})
            return res
        
        res=dic_arg.copy()
        if side=='buy':
            compute_price= lambda DF,K,x,Pt_T,delta: DF*np.maximum(x-K,0)*np.sum(delta*Pt_T[:,1:],axis=1)
        elif side=='sell':
            compute_price= lambda DF,K,x,Pt_T,delta: DF*np.maximum(K-x,0)*np.sum(delta*Pt_T[:,1:],axis=1)
        else: 
            raise ValueError(' Invalid input as side {side}')

        undl=[None]*len(self.pay_dates)
        for i,d in enumerate(self.call_dates):
            idx=Functions.find_idx(self.fix_dates,d)
            undl[idx]=compute_price(dic_arg["DF"][i],coupon,dic_arg["swap"][i],
                                    dic_arg["Pt_T"][i],dic_arg["delta"][i])
        
        res.update({'x':coupon,
                    'undl':undl})
        return res
