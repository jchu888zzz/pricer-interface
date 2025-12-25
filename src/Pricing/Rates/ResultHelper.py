
import numpy as np
import QuantLib as ql
from scipy.optimize import minimize

from Pricing.Rates.Payoffs import Base
from Pricing.Rates.Payoffs.Callable import CallableFeature
from collections import Counter
import typing

def organize_structure_table(contract,ZC) -> dict:

    res={'Payment Dates':contract.pay_dates}
    if hasattr(contract,'fwds'):
        res.update({'Model Forward':contract.fwds})
    res.update({
            'Early Redemption Proba':contract.proba_recall,
            'Cash Flows':contract.res_coupon,
            'Zero Coupon':ZC})
    
    return {"table":res}

def organize_funding_table(funding_leg,ZC:np.ndarray)-> dict:

    res={'Payment Dates':funding_leg.pay_dates,
        'Model Forward':np.mean(funding_leg.fwds,axis=1),
        'Proba':funding_leg.proba,
        'Cash Flows': funding_leg.coupons,
        'Zero Coupon':ZC}

    return res

def spread_interp(risky_curve,cal=ql.Thirty360(ql.Thirty360.BondBasis)) -> typing.Callable:
    entity=risky_curve.model.entity
    today=ql.Date.todaysDate()
    grid=[cal.yearFraction(today,today+ql.Period(p)) for p in entity.tenors]

    return lambda x: np.interp(x,grid,entity.quotes,left=entity.quotes[0],right=entity.quotes[-1])

def get_funding_spread_early_redemption(risky_curve,tgrid:np.ndarray,proba:np.ndarray) ->np.ndarray:
    c=0.0004
    interp_spreads=spread_interp(risky_curve)(tgrid) 
    res = sum(proba*interp_spreads) - c
    return res

def get_funding_spread(risky_curve,T:float):
    return spread_interp(risky_curve)(T) 

def optimize_coupon(func_to_solve:typing.Callable):
    init=0.04
    bounds=[(0.01,.5)]
    opt=minimize(func_to_solve,x0=init,method='Nelder-Mead',bounds=bounds)
    return opt.x[0]


def compute_bond_price_callable(dic_prep:dict,risky_curve,risky:bool):
    
    contract=dic_prep['contract']
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky)

    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows=contract.compute_cashflows(dic_arg)

    if 'dic_arg_helper' in dic_prep.keys():
        deg=5
        dic_arg_helper=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg_helper'])
        regressions=CallableFeature.get_regression_for_bond_with_undl(contract,dic_arg_helper,deg)
        
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                deg,include_principal=True)
        
        contract.compute_cashflows(dic_arg)
        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        contract.res_coupon=np.mean(cashflows,axis=0)
        contract.proba_recall=contract.compute_recall_proba(stop_idxs)
        contract.res_capital=Base.compute_bond_measure_change(dic_arg['measure_change_factor'],
                                                                stop_idxs)

    contract.res_coupon=np.mean(cashflows,axis=0)
    prices=contract.res_coupon+contract.res_capital
    price=sum(prices*ZC)
    
    res=organize_structure_table(contract,ZC)
    res['Price']=price
    res["duration"]=sum(contract.proba_recall*contract.paygrid)
    res["coupon"]=contract.coupon
    res["funding_spread"]=get_funding_spread_early_redemption(risky_curve,
                                                                contract.paygrid,contract.proba_recall)
    return res

def compute_swap_price_callable(dic_prep:dict,risky_curve):
    
    contract=dic_prep['contract']
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)

    funding_leg=dic_prep['funding_leg']
    funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    
    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows=contract.compute_cashflows(dic_arg)

    if not 'dic_arg_helper' in dic_prep.keys():
        funding_leg.compute_values(contract.funding_spread)

    else:
        deg=5
        dic_arg_helper=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg_helper'])
        regressions=CallableFeature.get_regression_for_swap_with_undl(contract,dic_arg_helper,
                                                                        funding_leg,contract.funding_spread,
                                                                        deg)
        
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                deg,include_principal=False)

        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        contract.proba_recall=contract.compute_recall_proba(stop_idxs)
        
        funding_leg.compute_values_for_early_redemption(stop_idxs,contract.funding_spread)

    contract.res_coupon=np.mean(cashflows,axis=0)
    structure_price=sum(contract.res_coupon*ZC)
    funding_price=sum(funding_leg.coupons*funding_ZC)
    price=structure_price-funding_price

    return {'Structure':organize_structure_table(contract,ZC),
            'Funding':organize_funding_table(funding_leg,funding_ZC),
            'Price':price,
            'duration':sum(contract.proba_recall*contract.paygrid),
            'coupon':contract.coupon,
            'funding_spread':contract.funding_spread}

def solve_coupon_for_bond_callable(dic_prep:dict,risky_curve,risky:bool):
    
    contract=dic_prep['contract']
    target=contract.UF+contract.yearly_buffer*contract.paygrid[-1]
    
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky)

    if not 'dic_arg_helper' in dic_prep.keys():
        res_funding=get_funding_spread(risky_curve,contract.paygrid[-1])
        def func_to_solve(x:float):
            dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
            cashflows=contract.compute_cashflows(dic_arg)
            coupons=np.mean(cashflows,axis=0)
            return (sum((coupons+contract.res_capital)*ZC) - (1-target))**2

        res_coupon=optimize_coupon(func_to_solve)
        return res_coupon,res_funding
    else:
        deg=5
        def func_to_solve(x:float):
            dic_arg_helper=contract.update_arg_pricing(x,dic_prep['dic_arg_helper']) 
            regressions=CallableFeature.get_regression_for_bond_with_undl(contract,dic_arg_helper,
                                                                        deg)
            dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
            stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                    deg,include_principal=True)
            cashflows=contract.compute_cashflows(dic_arg)
            cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
            coupons=np.mean(cashflows,axis=0)
            capital=Base.compute_bond_measure_change(dic_arg['measure_change_factor'],
                                                        stop_idxs)
            return (sum((coupons+capital)*ZC) - (1-target))**2

        res_coupon=optimize_coupon(func_to_solve)
        dic_arg_helper=contract.update_arg_pricing(res_coupon,dic_prep['dic_arg_helper'])
        dic_arg=contract.update_arg_pricing(res_coupon,dic_prep['dic_arg']) 
        regressions=CallableFeature.get_regression_for_bond_with_undl(contract,dic_arg_helper,
                                                                        deg)
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                deg,include_principal=True)
        proba_recall=contract.compute_recall_proba(stop_idxs)
        res_funding=get_funding_spread_early_redemption(risky_curve,contract.paygrid,proba_recall)

        return res_coupon,res_funding


def solve_coupon_for_swap_callable(dic_prep:dict,risky_curve):
    
    contract=dic_prep['contract']
    funding_leg=dic_prep['funding_leg']
    target=contract.UF+contract.yearly_buffer*contract.paygrid[-1]


    funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)

    if not 'dic_arg_helper' in dic_prep.keys():
        res_spread=get_funding_spread(risky_curve,contract.paygrid[-1])
        # compute coupons and proba
        funding_leg.compute_values(res_spread) 
        funding_price=sum(funding_leg.coupons*funding_ZC)
        
        def func_to_solve(x:float):
            dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
            cashflows=contract.compute_cashflows(dic_arg)
            res_coupon=np.mean(cashflows,axis=0)
            structure_price=sum(res_coupon*ZC)
            return (structure_price-funding_price +target)**2
        
        res_coupon=optimize_coupon(func_to_solve)
        res_funding=get_funding_spread(risky_curve,contract.paygrid[-1])
        return res_coupon,res_funding
    
    else:
        deg=5

        def func_to_solve(x:float):
            dic_arg_helper=contract.update_arg_pricing(x,dic_prep['dic_arg_helper']) 
            dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
            #Compute spread based on bond's duration
            regressions=CallableFeature.get_regression_for_bond_with_undl(contract,dic_arg_helper,
                                                                            deg)
            stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                    deg,include_principal=True)
            proba_recall=contract.compute_recall_proba(stop_idxs)
            funding_spread=get_funding_spread_early_redemption(risky_curve,contract.paygrid,proba_recall)
            #Use spread to compute swap value
            regressions=CallableFeature.get_regression_for_swap_with_undl(contract,dic_arg_helper,
                                                                        funding_leg,funding_spread,
                                                                        deg)
        
            stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                    deg,include_principal=False)
            cashflows=contract.compute_cashflows(dic_arg)
            cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
            coupons=np.mean(cashflows,axis=0)
            structure_price=sum(coupons*ZC)
        
            funding_leg.compute_values_for_early_redemption(stop_idxs,funding_spread)
            funding_price=sum(funding_leg.coupons*funding_ZC)
            return (structure_price-funding_price +target)**2

        res_coupon=optimize_coupon(func_to_solve)
        dic_arg_helper=contract.update_arg_pricing(res_coupon,dic_prep['dic_arg_helper']) 
        dic_arg=contract.update_arg_pricing(res_coupon,dic_prep['dic_arg']) 

        #compute spread based on the coupon value 
        regressions=CallableFeature.get_regression_for_bond_with_undl(contract,dic_arg_helper,deg)
        stop_idxs=CallableFeature.compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                                deg,include_principal=True)
        proba_recall=contract.compute_recall_proba(stop_idxs)
        res_funding=get_funding_spread_early_redemption(risky_curve,contract.paygrid,proba_recall)

        return res_coupon,res_funding
