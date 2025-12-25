import numpy as np
import QuantLib as ql

from Pricing.Utilities import InputConverter
from Pricing.Rates.Payoffs import Base
from Pricing.Utilities import Functions
from Pricing.Rates import Funding,ResultHelper

def precomputation(calc_date:ql.Date,model,data:dict[str:str]):

    contract=TARN(data)
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                       cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=0)
    
    contract.compute_grid(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    dic_arg=Base.prep_undl(contract,model,data_rates,include_rates=True)
    
    measure_change_factor=np.array([Base.compute_measure_change_factor(model,dic_arg['rates'][i],t,contract.paygrid[-1]) 
                                        for i,t in enumerate(contract.paygrid) ])
    
    contract.fwds=[np.mean(x) for x in dic_arg['undl']]
    
    res={'contract':contract,
         'measure_change_factor':measure_change_factor,
         'dic_arg':dic_arg}

    if contract.structure_type!='Swap':
        return res
    else:
        dic_currency={'EUR':'3M','USD':'Overnight'}
        funding_leg=Funding.Leg(contract,dic_currency[contract.currency])
        funding_leg.precomputation(calc_date,model,data_rates)
    
        res.update({'funding_leg':funding_leg})
    return res 

def compute_bond_price(dic_prep:dict,risky_curve,risky:bool):
    
    contract=dic_prep['contract']
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky)

    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows,stop_idxs=contract.compute_cashflows(dic_arg,include_stop_idx=True)

    contract.res_coupon=np.mean(cashflows,axis=0)
    contract.proba_recall=contract.compute_recall_proba(stop_idxs)
    contract.res_capital=Base.compute_bond_measure_change(dic_prep['measure_change_factor'],
                                                              stop_idxs)
    contract.res_coupon=np.mean(cashflows,axis=0)
    prices=contract.res_coupon+contract.res_capital
    price=sum(prices*ZC)
    
    res=ResultHelper.organize_structure_table(contract,ZC)
    res["Price"]=price
    res["duration"]=sum(contract.proba_recall*contract.paygrid)
    res["coupon"]=contract.coupon
    res["funding_spread"]=ResultHelper.get_funding_spread_early_redemption(risky_curve,
                                                                           contract.paygrid,contract.proba_recall)
    return res

def compute_swap_price(dic_prep:dict,risky_curve):
    
    contract=dic_prep['contract']
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)

    funding_leg=dic_prep['funding_leg']
    funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    
    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows,stop_idxs=contract.compute_cashflows(dic_arg,include_stop_idx=True)
    contract.res_coupon=np.mean(cashflows,axis=0)
    contract.proba_recall=contract.compute_recall_proba(stop_idxs)

    funding_leg.compute_values_for_early_redemption(stop_idxs,contract.funding_spread)
    structure_price=sum(contract.res_coupon*ZC)
    funding_price=sum(funding_leg.coupons*funding_ZC)
    price=structure_price-funding_price

    return {'Structure':ResultHelper.organize_structure_table(contract,ZC),
            'Funding':ResultHelper.organize_funding_table(funding_leg,funding_ZC),
            'Price':price,
            'duration':sum(contract.proba_recall*contract.paygrid),
            'coupon':contract.coupon,
            'funding_spread':ResultHelper.get_funding_spread_early_redemption(risky_curve,
                                                                           contract.paygrid,contract.proba_recall)}


class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                                param_contract)
        if param_contract['structure_type']=='Bond':
            return compute_bond_price(dic_prep,prep_model['risky_curve'],risky=True)
        else:
            return compute_swap_price(dic_prep,prep_model['risky_curve'])

class TARN(Base.Payoff):

    def __init__(self,parameters:dict):
        self.get_common_parameters(parameters)
        self.coupon_lvl=InputConverter.set_param(parameters['coupon_level'],0)
        self.memory=False
        
        self.get_guaranteed_coupon_info(parameters)
        self.target=InputConverter.set_param(parameters['target'],0)
        self.call_dates=self.fix_dates
    
    
    def compute_cashflows(self,dic_arg:dict,include_stop_idx=False) -> np.ndarray:
        coupon=dic_arg['x']
        undl=dic_arg['undl']
        
        guaranteed_cashflows=np.zeros_like(undl.T)
        guaranteed_cashflows[:,:self.NC+1]=self.guaranteed_coupon

        cdt_cashflows=Base.compute_cdt_digit(undl,self.coupon_lvl,
                             self.infine,self.memory)*coupon
        cdt_cashflows[:,:self.NC+1]=False
        
        cashflows=guaranteed_cashflows+cdt_cashflows

        autocall_cdt=(np.cumsum(cashflows,axis=1) >=self.target )
        stop_idxs=[Functions.first_occ(np.array(x),True) for x in autocall_cdt]

        if include_stop_idx:
            return Base.adjust_to_stop_idxs(cashflows,stop_idxs,self.infine),stop_idxs

        return Base.adjust_to_stop_idxs(cashflows,stop_idxs,self.infine)
    
    def update_arg_pricing(self,coupon:float,dic_arg:dict) -> dict:
        res=dic_arg.copy()
        res.update({'x':coupon})
        return res

    


    