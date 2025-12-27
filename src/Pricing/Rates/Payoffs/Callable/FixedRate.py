import numpy as np
import QuantLib as ql
from Pricing.Rates.Payoffs import  Base
from Pricing.Rates import Funding
from . import CallableFeature

def precomputation(calc_date:ql.Date,model,data:dict[str:str],risky_curve,risky:bool):

    contract=FixedRate(data)

    dic_currency={'EUR':'3M','USD':'Overnight'}

    contract.compute_grid(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    contract.compute_funding_adjustment(calc_date)
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                                    cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=0)
    dic_arg=model.compute_prep_for_swaption_from_rates(data_rates,contract.fixgrid,
                                                        contract.paygrid[-1],contract.call_idxs,
                                                        include_rates=True)

    res=dict()
    if contract.structure_type=='Swap':
        funding_leg=Funding.Leg(contract,dic_currency[contract.currency])
        funding_leg.precomputation(calc_date,model,data_rates)
        res.update({'funding_leg':funding_leg})


    data_rates_helper=model.generate_rates(calc_date,contract.pay_dates[-1],cal=ql.Thirty360(ql.Thirty360.BondBasis),
                                    Nbsimu=1000,seed=42)
        
    dic_arg_helper=model.compute_prep_for_swaption_from_rates(data_rates_helper,contract.fixgrid,
                                                                contract.paygrid[-1],contract.call_idxs,
                                                                include_rates=True)

    df_cont_helper=[None]*len(contract.call_idxs)  
    df_exercise=[None]*len(contract.call_idxs)
    for i,idx in enumerate(contract.call_idxs):
        t=contract.fixgrid[idx]
        df_exercise[i]=model.compute_discount_factor_from_rates(dic_arg['rates'][idx],t,
                                                                            contract.paygrid[idx])*model.DF(contract.fixgrid[idx])
        df_cont_helper[i]=model.compute_discount_factor_from_rates(dic_arg_helper['rates'][idx],t,
                                                                            contract.paygrid[idx+1:])*model.DF(contract.fixgrid[idx])
        if risky:
            df_exercise[i]*=risky_curve.adjustment(t,contract.paygrid[idx])
            df_cont_helper[i]*=np.array([risky_curve.adjustment(t,T)  
                                            for T in contract.paygrid[idx+1:] ])

    measure_change_factor=np.array([Base.compute_measure_change_factor(model,dic_arg['rates'][i],t,contract.paygrid[-1]) 
                                    for i,t in enumerate(contract.paygrid) ])
    
    dic_arg.update({'df_exercise':df_exercise,
                    'measure_change_factor':measure_change_factor})

    dic_arg_helper.update({'df_continuation':df_cont_helper})

    res.update({'contract':contract,
                'dic_arg_helper':dic_arg_helper,
                'dic_arg':dic_arg})
    return res

def compute_price(dic_prep:dict,risky_curve):
    return CallableFeature.compute_price(dic_prep,risky_curve)

def solve_coupon(dic_prep:dict,risky_curve):
    return CallableFeature.solve_coupon(dic_prep,risky_curve)

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

class FixedRate(Base.Payoff) :

    def __init__(self,parameters:dict):
        self.fixing_type='in arrears'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)    

    def compute_cashflows(self,dic_arg:dict):
        coupon=dic_arg['x']
        nb_simu=dic_arg['rates'].shape[1]
        res=coupon*self.delta

        if self.infine:
            return np.tile(np.cumsum(res),(nb_simu,1))
        else:
            return np.tile(res,(nb_simu,1))
        
    def update_arg_pricing(self,coupon:float,dic_arg:dict,side:str='sell') -> dict:
        res=dic_arg.copy()
        if side=='buy':
            compute_price= lambda DF,K,x,Pt_T,delta: DF*np.maximum(x-K,0)*np.sum(delta*Pt_T[:,1:],axis=1)
        elif side=='sell':
            compute_price= lambda DF,K,x,Pt_T,delta: DF*np.maximum(K-x,0)*np.sum(delta*Pt_T[:,1:],axis=1)
        else: 
            raise ValueError(' Invalid input as side {side}')

        undl=[None]*len(self.paygrid)
        for i,idx in enumerate(self.call_idxs):
            undl[idx]=compute_price(dic_arg["DF"][i],coupon,dic_arg["swap"][i],
                                    dic_arg["Pt_T"][i],dic_arg["delta"][i])
        
        res.update({'x':coupon,
                    'undl':undl})
        return res