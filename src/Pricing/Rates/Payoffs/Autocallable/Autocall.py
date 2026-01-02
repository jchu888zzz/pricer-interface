import numpy as np
import QuantLib as ql

import Pricing.Utilities.InputConverter as InputConverter
import Pricing.Utilities.Functions as Functions
import Pricing.Rates.Payoffs.Base as Base
from Pricing.Rates import Funding

def precomputation(calc_date:ql.Date,model,data:dict[str:str]):

    contract=Autocall(data)
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                       cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=0)
    
    contract._update(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    contract.compute_funding_adjustment(calc_date)
    dic_arg=Base.prep_undl(contract,model,data_rates,include_rates=True)
    stop_idxs=contract.compute_stop_idxs(dic_arg['undl'])
    
    calendar=model.curve.calendar
    contract.paygrid=np.array([calendar.yearFraction(calc_date,d) for d in contract.pay_dates])
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
        funding_leg=Funding.Leg(contract,contract.currency)
        funding_leg.precomputation(calc_date,model,data_rates)
    
        res.update({'funding_leg':funding_leg})
    return res    


def compute_price(dic_prep:dict,risky_curve):

    res=dict()

    contract=dic_prep['contract']
    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows=contract.compute_cashflows(dic_arg)
    contract.res_coupon=np.mean(cashflows,axis=0)
    
    if contract.structure_type=="Bond":
        zc=risky_curve.discount_factor(contract.pay_dates,risky=True)
        price=sum((contract.res_coupon+contract.res_capital)*zc)
    
    elif contract.structure_type=="Swap":
        funding_leg=dic_prep['funding_leg']
        funding_ZC=risky_curve.discount_factor(funding_leg.pay_dates,risky=False)
        funding_leg.compute_values_for_early_redemption(dic_arg['stop_idxs'],contract.funding_spread)
        funding_price=sum(funding_leg.coupons*funding_ZC)

        zc=risky_curve.discount_factor(contract.pay_dates,risky=False)
        structure_price=sum(contract.res_coupon*zc)
        
        price=structure_price-funding_price
        res['funding_table']=Base.organize_funding_table(funding_leg,funding_ZC)
    else:
        raise ValueError(f"{contract.structure_type} not recognized")

    res["table"]=Base.organize_structure_table(contract,zc)
    res["price"]=price
    res["duration"]=contract.duration
    res["coupon"]=contract.coupon
    res["funding_spread"]=Base.get_funding_spread_early_redemption(risky_curve,
                                                                contract.pay_dates,contract.proba_recall,
                                                                contract.funding_adjustment)
    return res

def solve_coupon(dic_prep:dict,risky_curve):

    contract=dic_prep['contract']
    target=contract.UF+contract.yearly_buffer*contract.paygrid[-1]
    res_funding=Base.get_funding_spread_early_redemption(risky_curve,
                                                        contract.pay_dates,contract.proba_recall,
                                                        contract.funding_adjustment)

    if contract.structure_type=="Bond":
        zc=risky_curve.discount_factor(contract.pay_dates,risky=True)
        def func_to_solve(x:float):
            dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
            cashflows=contract.compute_cashflows(dic_arg)
            coupons=np.mean(cashflows,axis=0)
            return (sum((coupons+contract.res_capital)*zc) - (1-target))**2

        res_coupon=Base.optimize_coupon(func_to_solve)
        return res_coupon,res_funding
    
    if contract.structure_type=="Swap":
        funding_leg=dic_prep['funding_leg']
        funding_zc=risky_curve.discount_factor(funding_leg.pay_dates,risky=False)
        funding_leg.compute_values_for_early_redemption(dic_prep['dic_arg']['stop_idxs'],res_funding)
        funding_price=sum(funding_leg.coupons*funding_zc)

        zc=risky_curve.Discount_Factor(contract.pay_dates,risky=False)
        def func_to_solve(x:float):
            dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
            cashflows=contract.compute_cashflows(dic_arg)
            res_coupon=np.mean(cashflows,axis=0)
            structure_price=sum(res_coupon*zc)
            return (structure_price-funding_price +target)**2        
        
        res_coupon=Base.optimize_coupon(func_to_solve)
        return res_coupon,res_funding

    raise ValueError(f"{contract.structure_type} not recognized")

class Process :
    def compute_price(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                                param_contract)
        return compute_price(dic_prep,prep_model['risky_curve'])
        
    def solve_coupon(prep_model:dict,param_contract:dict):
        dic_prep=precomputation(prep_model['calc_date'],prep_model['model'],
                         param_contract)
        
        def update_dic_prep(coupon,spread) ->dict:
            dic_prep_new=dic_prep.copy()
            dic_prep_new['contract'].coupon=coupon
            dic_prep_new['contract'].funding_spread=spread
            return dic_prep_new

        coupon,spread=solve_coupon(dic_prep,prep_model['risky_curve'])
        dic_prep_new=update_dic_prep(coupon,spread)
        return compute_price(dic_prep_new,prep_model['risky_curve'])


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
        non_call=len(self.fix_dates) -(len(self.call_dates)+1)
        autocall_cdt[:,:non_call]=0 # set period before call to 0
        return Functions.first_occ_vec(autocall_cdt, True) 
    
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