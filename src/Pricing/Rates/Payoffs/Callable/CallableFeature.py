import numpy as np
import QuantLib as ql
import typing 
import scipy
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


import Pricing.Utilities.Functions as Functions
import Pricing.Rates.Payoffs.Base as Base
from Pricing.Rates import Funding


def get_polynomial_basis(undl:np.ndarray,deg:int):
    basis=undl.copy()
    for j in range(2,deg+1):
        basis=np.c_[undl**j,basis]
    return basis

def get_laguerre_basis(undl:np.ndarray,deg:int):
    basis=scipy.special.eval_laguerre(1,undl)
    for i in range(2,deg+1):
        basis=np.c_[scipy.special.eval_laguerre(i,undl),basis]
    return basis


def retrieve_func_basis(deg:int,option="laguerre"):
    if option=='laguerre':
        return lambda x : get_laguerre_basis(x,deg)
    elif option=='polynomial':
        return lambda x : get_polynomial_basis(x,deg)
    else:
        raise ValueError(f'{option} not implemented')

def get_regression_for_bond_with_undl(contract,dic_arg:dict,
                                      deg:int,option='laguerre') -> list[RandomForestRegressor]:
    """ simulation rates shape: (len(tgrid),nb simu) """
    
    func_basis=retrieve_func_basis(deg,option=option)

    nbsimu=dic_arg['rates'].shape[1]
    df_continuation=dic_arg['df_continuation']
    cashflows=contract.compute_cashflows(dic_arg)
    stop_idxs=[len(contract.paygrid)-1]*nbsimu
    res=[]

    for idx,discount_factor in zip(reversed(contract.call_idxs),reversed(df_continuation)):
        adjusted_cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        Y=np.sum(adjusted_cashflows[:,idx+1:]*discount_factor,axis=1)
        Y+= np.array([x[k-(idx+1)] for x,k in zip(discount_factor,stop_idxs)])
        
        basis=func_basis(dic_arg['undl'][idx])
        #reg=RandomForestRegressor()
        reg=KNeighborsRegressor(n_neighbors=5)
        reg.fit(basis,Y)
        res.insert(0,reg)

    return res

def get_regression_for_swap_with_undl(contract,dic_arg:dict,fund_leg,
                                      spread:float,deg:int,option='laguerre') -> list[RandomForestRegressor]:
    
    func_basis=retrieve_func_basis(deg,option=option)

    nbsimu=dic_arg['rates'].shape[1]
    cashflows=contract.compute_cashflows(dic_arg)

    def compute_fund_price(cf:np.ndarray,df:np.ndarray):
        if cf.size==0:
            return 0
        else:
            return sum(cf*df)

    stop_idxs=[len(contract.paygrid)-1]*nbsimu
    res=[]

    fund_cf=fund_leg.compute_cashflows(spread)
    for idx,Pt_T,fund_Pt_T in zip(reversed(contract.call_idxs),
                                  reversed(dic_arg['df_continuation']),
                                  reversed(fund_leg.df_continuation)):

        adjusted_cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        structure_price=np.sum(adjusted_cashflows[:,idx+1:]*Pt_T,axis=1)

        fundstop_idxs=[Functions.find_idx(fund_leg.pay_dates, contract.pay_dates[j]) for j in stop_idxs]
        fund_idx1=Functions.find_idx(fund_leg.pay_dates,contract.pay_dates[idx])

        fund_price=np.array([compute_fund_price(fund_cf[j,fund_idx1:fund_idx2],fund_Pt_T[j,:fund_idx2-fund_idx1]) 
                            for j,fund_idx2 in enumerate(fundstop_idxs)])
        
        Y=structure_price -fund_price
        
        basis=func_basis(dic_arg['undl'][idx])
        #reg=RandomForestRegressor()
        reg=KNeighborsRegressor(n_neighbors=5)
        reg.fit(basis,Y)
        res.insert(0,reg)

    return res

def compute_stop_idxs_with_undl(contract,regressions:list[typing.Callable],dic_arg:dict,
                                deg:int,include_principal:bool,option='laguerre') ->list[int]:
    
    func_basis=retrieve_func_basis(deg,option=option)
        
    nbsimu=dic_arg['rates'].shape[1]
    res=[len(contract.paygrid)-1]*nbsimu
    stop_idxs=[]

    cashflows=contract.compute_cashflows(dic_arg)

    for i,idx in enumerate(contract.call_idxs):   
        Exercise_value=dic_arg['df_exercise'][i]*(include_principal+contract.infine*cashflows[:,idx])
        basis=func_basis(dic_arg['undl'][idx])
        decision=regressions[i].predict(basis)>Exercise_value
        new_stop_idxs=[ j for j,b in enumerate(decision) if b==1 and j not in stop_idxs]
        stop_idxs+=new_stop_idxs
        res=[ idx if j in new_stop_idxs else x for j,x in enumerate(res) ]
        
    return res

def prep_callable_contract(calc_date:ql.Date,contract,model,risky_curve,risky:bool) :

    dic_currency={'EUR':'3M','USD':'Overnight'}

    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                                    cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000)
    
    contract.compute_grid(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    contract.compute_funding_adjustment(calc_date)
    dic_arg=Base.prep_undl(contract,model,data_rates,include_rates=True)
    if contract.hasunderlying :
        contract.fwds=[np.mean(x) for x in dic_arg['undl']]
    
    res=dict()

    if contract.structure_type=='Swap':
        funding_leg=Funding.Leg(contract,dic_currency[contract.currency])
        funding_leg.precomputation(calc_date,model,data_rates)
        res.update({'funding_leg':funding_leg})

    if not hasattr(contract,'call_dates'):

        contract.proba_recall=np.zeros_like(contract.paygrid)
        contract.proba_recall[-1]=1
        contract.res_capital=contract.proba_recall
        contract.duration=contract.paygrid[-1]
        res.update({'contract':contract,
                    'dic_arg':dic_arg})

        return res

    else:
        data_rates_helper=model.generate_rates(calc_date,contract.pay_dates[-1],cal=ql.Thirty360(ql.Thirty360.BondBasis),
                                    Nbsimu=1000,seed=42)
        
        dic_arg_helper=Base.prep_undl(contract,model,data_rates_helper,include_rates=True)
        
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
    """
    Compute price for callable bond or swap.
    """
    
    contract=dic_prep['contract']
    
    # Check if this is a swap or bond
    is_swap='funding_leg' in dic_prep.keys()
    
    if is_swap:
        ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)
        funding_leg=dic_prep['funding_leg']
        funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
    else:
        ZC=risky_curve.Discount_Factor(contract.paygrid,risky=True)
    
    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows=contract.compute_cashflows(dic_arg)

    #Process if callable
    if 'dic_arg_helper' in dic_prep.keys():
        deg=5
        dic_arg_helper=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg_helper'])
        
        if is_swap:
            regressions=get_regression_for_swap_with_undl(contract,dic_arg_helper,
                                                        funding_leg,contract.funding_spread,deg)
            include_principal=False
        else:
            regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg)
            include_principal=True
        
        stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                            deg,include_principal=include_principal)

        contract.compute_cashflows(dic_arg)
        cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        contract.res_coupon=np.mean(cashflows,axis=0)
        contract.proba_recall=contract.compute_recall_proba(stop_idxs)
        
        if not is_swap:
            contract.res_capital=Base.compute_bond_measure_change(dic_arg['measure_change_factor'],
                                                                    stop_idxs)
        else:
            funding_leg.compute_values_for_early_redemption(stop_idxs,contract.funding_spread)
    else:
        if is_swap:
            funding_leg.compute_values(contract.funding_spread)

    contract.res_coupon=np.mean(cashflows,axis=0)
    
    if is_swap:
        structure_price=sum(contract.res_coupon*ZC)
        funding_price=sum(funding_leg.coupons*funding_ZC)
        price=structure_price-funding_price
    else:
        prices=contract.res_coupon+contract.res_capital
        price=sum(prices*ZC)
    
    res=dict()
    res['table']=Base.organize_structure_table(contract,ZC)
    res['price']=price
    res["duration"]=sum(contract.proba_recall*contract.paygrid)
    res["coupon"]=contract.coupon
    res["funding_spread"]=Base.get_funding_spread_early_redemption(risky_curve,
                                                                contract.paygrid,contract.proba_recall,
                                                                contract.funding_adjustment)
    
    if is_swap:
        res['funding_table']=Base.organize_funding_table(funding_leg,funding_ZC)
    
    return res

def solve_coupon(dic_prep:dict,risky_curve):
    """
    Solve for optimal coupon for callable bond or swap.
    For bonds: risky=True/False determines discount factor
    For swaps: risky=False (ignored), includes funding leg pricing
    """
    
    contract=dic_prep['contract']
    target=contract.UF+contract.yearly_buffer*contract.paygrid[-1]
    
    # Check if this is a swap (has funding_leg) or bond
    is_swap='funding_leg' in dic_prep.keys()
    
    if is_swap:
        funding_leg=dic_prep['funding_leg']
        funding_ZC=risky_curve.Discount_Factor(funding_leg.paygrid,risky=False)
        ZC=risky_curve.Discount_Factor(contract.paygrid,risky=False)
    else:
        ZC=risky_curve.Discount_Factor(contract.paygrid,risky=True)

    if not 'dic_arg_helper' in dic_prep.keys():
        if is_swap:
            res_spread=Base.get_funding_spread(risky_curve,contract.paygrid[-1],contract.funding_adjustment)
            funding_leg.compute_values(res_spread) 
            funding_price=sum(funding_leg.coupons*funding_ZC)
            
            def func_to_solve(x:float):
                dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
                cashflows=contract.compute_cashflows(dic_arg)
                res_coupon=np.mean(cashflows,axis=0)
                structure_price=sum(res_coupon*ZC)
                return (structure_price-funding_price +target)**2
            
            res_coupon=Base.optimize_coupon(func_to_solve)
            res_funding=Base.get_funding_spread(risky_curve,contract.paygrid[-1],contract.funding_adjustment)
        else:
            res_funding=Base.get_funding_spread(risky_curve,contract.paygrid[-1],contract.funding_adjustment)
            def func_to_solve(x:float):
                dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
                cashflows=contract.compute_cashflows(dic_arg)
                coupons=np.mean(cashflows,axis=0)
                return (sum((coupons+contract.res_capital)*ZC) - (1-target))**2

            res_coupon=Base.optimize_coupon(func_to_solve)
        
        return res_coupon,res_funding
    
    else:
        deg=5
        if is_swap:
            def func_to_solve(x:float):
                dic_arg_helper=contract.update_arg_pricing(x,dic_prep['dic_arg_helper']) 
                dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
                #Compute spread based on bond's duration
                regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg)
                stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                    deg,include_principal=True)
                proba_recall=contract.compute_recall_proba(stop_idxs)
                funding_spread=Base.get_funding_spread_early_redemption(risky_curve,contract.paygrid,proba_recall,contract.funding_adjustment)
                #Use spread to compute swap value
                regressions=get_regression_for_swap_with_undl(contract,dic_arg_helper,
                                                            funding_leg,funding_spread,deg)
            
                stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                    deg,include_principal=False)
                cashflows=contract.compute_cashflows(dic_arg)
                cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
                coupons=np.mean(cashflows,axis=0)
                structure_price=sum(coupons*ZC)
            
                funding_leg.compute_values_for_early_redemption(stop_idxs,funding_spread)
                funding_price=sum(funding_leg.coupons*funding_ZC)
                return (structure_price-funding_price +target)**2
        else:
            def func_to_solve(x:float):
                dic_arg_helper=contract.update_arg_pricing(x,dic_prep['dic_arg_helper']) 
                regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,
                                                            deg)
                dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
                stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                    deg,include_principal=True)
                cashflows=contract.compute_cashflows(dic_arg)
                cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
                coupons=np.mean(cashflows,axis=0)
                capital=Base.compute_bond_measure_change(dic_arg['measure_change_factor'],
                                                            stop_idxs)
                return (sum((coupons+capital)*ZC) - (1-target))**2

        res_coupon=Base.optimize_coupon(func_to_solve)
        
        dic_arg_helper=contract.update_arg_pricing(res_coupon,dic_prep['dic_arg_helper']) 
        dic_arg=contract.update_arg_pricing(res_coupon,dic_prep['dic_arg']) 

        if is_swap:
            #compute spread based on the coupon value 
            regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg)
            stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                deg,include_principal=True)
            proba_recall=contract.compute_recall_proba(stop_idxs)
            res_funding=Base.get_funding_spread_early_redemption(risky_curve,contract.paygrid,proba_recall,contract.funding_adjustment)
        else:
            regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,
                                                        deg)
            stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                deg,include_principal=True)
            proba_recall=contract.compute_recall_proba(stop_idxs)
            res_funding=Base.get_funding_spread_early_redemption(risky_curve,contract.paygrid,
                                                                proba_recall,contract.funding_adjustment)

        return res_coupon,res_funding
