import numpy as np
import QuantLib as ql

import Pricing.Utilities.InputConverter as InputConverter
import Pricing.Utilities.Functions as Functions
import Pricing.Rates.Payoffs.Base as Base
from Pricing.Rates import Funding,ResultHelper

def precomputation(calc_date:ql.Date,model,data:dict[str:str]):

    contract=Autocall(data)
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                       cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=0)
    
    contract.compute_grid(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    dic_arg=Base.prep_undl(contract,model,data_rates,include_rates=True)
    stop_idxs=contract.compute_stop_idxs(dic_arg['undl'])
    
    measure_change_factor=np.array([Base.compute_measure_change_factor(model,dic_arg['rates'][i],t,contract.paygrid[-1]) 
                                        for i,t in enumerate(contract.paygrid) ])
    
    contract.proba_recall=contract.compute_recall_proba(stop_idxs)
    contract.res_capital=Base.compute_bond_measure_change(measure_change_factor,stop_idxs)
    contract.duration=sum(contract.compute_recall_proba(stop_idxs)*contract.paygrid)
    contract.fwds=[np.mean(x) for x in dic_arg['undl']]
    
    dic_arg.update({'stop_idxs':stop_idxs})
    res={'contract':contract,
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
    cashflows=contract.compute_cashflows(dic_arg)

    contract.res_coupon=np.mean(cashflows,axis=0)
    prices=contract.res_coupon+contract.res_capital
    price=sum(prices*ZC)
    
    res=ResultHelper.organize_structure_table(contract,ZC)
    res["Price"]=price
    res["duration"]=contract.duration
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
    cashflows=contract.compute_cashflows(dic_arg)
    contract.res_coupon=np.mean(cashflows,axis=0)

    funding_leg.compute_values_for_early_redemption(dic_arg['stop_idxs'],contract.funding_spread)
    structure_price=sum(contract.res_coupon*ZC)
    funding_price=sum(funding_leg.coupons*funding_ZC)
    price=structure_price-funding_price

    return {'Structure':ResultHelper.organize_structure_table(contract,ZC),
            'Funding':ResultHelper.organize_funding_table(funding_leg,funding_ZC),
            'Price':price,
            'duration':contract.duration,
            'coupon':contract.coupon,
            'funding_spread':ResultHelper.get_funding_spread_early_redemption(risky_curve,
                                                                           contract.paygrid,contract.proba_recall)}

def solve_coupon_for_swap(dic_prep:dict,risky_curve):
    
    contract=dic_prep['contract']
    funding_leg=dic_prep['funding_leg']
    target=contract.UF+contract.yearly_buffer*contract.paygrid[-1]

    funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)

    res_funding=ResultHelper.get_funding_spread_early_redemption(risky_curve,contract.paygrid,
                                                                 contract.proba_recall)
    
    funding_leg.compute_values_for_early_redemption(dic_prep['dic_arg']['stop_idxs'],res_funding)
    funding_price=sum(funding_leg.coupons*funding_ZC)

    def func_to_solve(x:float):
        dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
        cashflows=contract.compute_cashflows(dic_arg)
        res_coupon=np.mean(cashflows,axis=0)
        structure_price=sum(res_coupon*ZC)
        return (structure_price-funding_price +target)**2
    
    res_coupon=ResultHelper.optimize_coupon(func_to_solve)
    return res_coupon,res_funding

def solve_coupon_for_bond(dic_prep:dict,risky_curve,risky:bool):
    
    contract=dic_prep['contract']
    target=contract.UF+contract.yearly_buffer*contract.paygrid[-1]
    
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky)
    res_funding=ResultHelper.get_funding_spread_early_redemption(risky_curve,contract.paygrid,contract.proba_recall)
    def func_to_solve(x:float):
        dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
        cashflows=contract.compute_cashflows(dic_arg)
        coupons=np.mean(cashflows,axis=0)
        return (sum((coupons+contract.res_capital)*ZC) - (1-target))**2
        
    res_coupon=ResultHelper.optimize_coupon(func_to_solve)
    return res_coupon,res_funding


class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                                param_contract)
        if param_contract['structure_type']=='Bond':
            return compute_bond_price(dic_prep,prep_model['risky_curve'],risky=True)
        else:
            return compute_swap_price(dic_prep,prep_model['risky_curve'])
        
    def solve_coupon(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                         param_contract)
        
        def update_dic_prep(coupon,spread) ->dict:
            dic_prep_new=dic_prep.copy()
            dic_prep_new['contract'].coupon=coupon
            dic_prep_new['contract'].funding_spread=spread
            return dic_prep_new

        if param_contract['structure_type']=='Bond':
            coupon,spread=solve_coupon_for_bond(dic_prep,prep_model['risky_curve'],risky=True)
            dic_prep_new=update_dic_prep(coupon,spread)
            return compute_bond_price(dic_prep_new,prep_model['risky_curve'],risky=True)
        else:
            coupon,spread=solve_coupon_for_swap(dic_prep,prep_model['risky_curve'])
            dic_prep_new=update_dic_prep(coupon,spread)
            return compute_swap_price(dic_prep_new,prep_model['risky_curve'])


class Autocall(Base.Payoff):

    def __init__(self,parameters:dict[str:str]):
        self.get_common_parameters(parameters)
        self.coupon_lvl=InputConverter.set_param(parameters['coupon_level'],0)
        self.autocall_lvl=InputConverter.set_param(parameters['autocall_level'],0)
        self.get_memory_effect(parameters)
        if 'call_dates' in parameters.keys():
            self.call_dates=parameters['call_dates']
            return
        if 'NC' in parameters.keys():
            non_call=max(int(parameters['NC'])-1,0)
            self.call_dates=self.fix_dates[non_call:-1]

    def compute_stop_idxs(self,undl:np.ndarray) -> list:
        """ undl shape (nb time steps, nb simu)"""
        autocall_cdt=(undl.T <=self.autocall_lvl)
        autocall_cdt[:,:self.call_idxs[0]]=0 # set period before call to 0
        return [Functions.first_occ(np.array(x),True) for x in autocall_cdt]
        
    def compute_cashflows(self,dic_arg:dict) -> np.ndarray:
        """ dic_arg must contain necessary arguments for pricing cashflows """
        coupon=dic_arg['x']
        undl=dic_arg['undl']
        stop_idxs=dic_arg['stop_idxs']
        cashflow_cdt=Base.compute_cdt_digit(undl,self.coupon_lvl,
                             self.infine,self.memory)
        cashflow_cdt=Base.adjust_to_stop_idxs(cashflow_cdt,stop_idxs,self.infine)

        return coupon*cashflow_cdt
    
    def update_arg_pricing(self,coupon:float,dic_arg:dict) -> dict:
        res=dic_arg.copy()
        res.update({'x':coupon})
        return res