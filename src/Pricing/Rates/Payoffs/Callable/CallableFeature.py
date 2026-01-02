import numpy as np
import QuantLib as ql
import scipy
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import Pricing.Utilities.Functions as Functions
import Pricing.Rates.Payoffs.Base as Base
from Pricing.Rates import Funding

def prep_discount_factor_from_rates(contract,model,risky_curve
                                    ,dic_arg:dict,risky:bool):
    zc_cont=[None]*len(contract.call_dates)  
    zc_exec=[None]*len(contract.call_dates)
    
    calendar=risky_curve.calendar
    calc_date=risky_curve.calc_date

    for i,d in enumerate(contract.call_dates):
        idx=Functions.find_idx(contract.fix_dates,d)
        t_fix=calendar.yearFraction(calc_date,contract.fix_dates[idx])
        t_cont=np.array([calendar.yearFraction(calc_date,d) for d in contract.pay_dates[idx:]] )
        zc_exec[i]=model.compute_discount_factor_from_rates(dic_arg['rates'][idx],t_fix,
                                                                t_cont[0]).ravel()
        zc_cont[i]=model.compute_discount_factor_from_rates(dic_arg['rates'][idx],t_fix,
                                                                    t_cont[1:])

        if risky:
            zc_exec[i]*=risky_curve.adjustment(t_fix,t_cont[0])
            zc_cont[i]*=np.array([risky_curve.adjustment(t_fix,t) for t in t_cont[1:] ])

    dic_arg['zc_exercise']=zc_exec
    dic_arg['zc_continuation']=zc_cont    
    return dic_arg
    
def prep_callable_contract(calc_date:ql.Date,contract,model,risky_curve,risky:bool) :

    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                                    cal=ql.Actual360(),Nbsimu=10000)
    
    contract._update(calc_date,cal=ql.Thirty360(ql.Thirty360.BondBasis))
    contract.compute_funding_adjustment(calc_date)
    dic_arg=Base.prep_undl(contract,model,data_rates,include_rates=True)
    prep_discount_factor_from_rates(contract,model,risky_curve,dic_arg,risky)

    if contract.hasunderlying :
        #contract.fwds=[np.mean(x) for x in dic_arg['undl']]
        contract.fwds=np.mean(dic_arg['undl'],axis=1)
    res=dict()

    if contract.structure_type=='Swap':
        funding_leg=Funding.Leg(contract,contract.currency)
        funding_leg.precomputation(calc_date,model,data_rates)
        res.update({'funding_leg':funding_leg})

    contract.paygrid=np.array([risky_curve.calendar.yearFraction(calc_date,d) for d in contract.pay_dates ])
    if not hasattr(contract,'call_dates'):
        contract.proba_recall=np.zeros_like(contract.pay_dates)
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
        prep_discount_factor_from_rates(contract,model,risky_curve,dic_arg_helper,risky)
        
        measure_change_factor=np.array([Base.compute_measure_change_factor(model,dic_arg['rates'][i],t,contract.paygrid[-1]) 
                                        for i,t in enumerate(contract.paygrid) ])
        
        dic_arg['measure_change_factor']=measure_change_factor
        res.update({'contract':contract,
                    'dic_arg_helper':dic_arg_helper,
                    'dic_arg':dic_arg})
        return res

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

MAPPING_BASIS={'laguerre':lambda x,deg : get_laguerre_basis(x,deg),
                'polynomial': lambda x,deg :get_polynomial_basis(x,deg) }

def get_regression_for_bond_with_undl(contract,dic_arg:dict,
                                        deg:int,regressor_class:KNeighborsRegressor | Ridge,
                                        basis_option='polynomial') -> list[KNeighborsRegressor | Ridge]:
    """ simulation rates shape: (len(tgrid),nb simu) """

    nbsimu=dic_arg['rates'].shape[1]
    cashflows=contract.compute_cashflows(dic_arg)
    stop_idxs=[len(contract.pay_dates)-1]*nbsimu
    res=[]
    
    for d,df_cont,df_exercise in zip(reversed(contract.call_dates),
                                    reversed(dic_arg['zc_continuation']),
                                    reversed(dic_arg['zc_exercise'])):
        
        idx=Functions.find_idx(contract.fix_dates,d)
        adjusted_cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        #Continuation value
        continuation_value=np.sum(adjusted_cashflows[:,idx+1:]*df_cont,axis=1)
        #Add capital at stopping time
        continuation_value+= np.array([x[k-(idx+1)] for x,k in zip(df_cont,stop_idxs)])

        basis=MAPPING_BASIS.get(basis_option)(dic_arg['undl'][idx],deg)
        reg=clone(regressor_class)
        reg.fit(basis,continuation_value)
        res.insert(0,reg)
        exercise_value = df_exercise * (1 + contract.infine * cashflows[:, idx])
        prediction = reg.predict(basis)
        decision = prediction>exercise_value

        #Update stop idxs based on decision
        stop_idxs= [idx if d else old_value for d,old_value in zip(decision,stop_idxs) ]

    return res

def get_regression_for_swap_with_undl(contract,dic_arg:dict,fund_leg,
                                        spread:float,deg:int,
                                        regressor_class:KNeighborsRegressor | Ridge,
                                        basis_option='polynomial') -> list[KNeighborsRegressor | Ridge]:
    
    nbsimu=dic_arg['rates'].shape[1]
    cashflows=contract.compute_cashflows(dic_arg)

    def compute_fund_price(cf:np.ndarray,df:np.ndarray):
        if cf.size==0:
            return 0
        else:
            return sum(cf*df)

    stop_idxs=[len(contract.pay_dates)-1]*nbsimu
    res=[]

    fund_cf=fund_leg.compute_cashflows(spread)

    for d,Pt_T,fund_Pt_T in zip(reversed(contract.call_dates),
                                    reversed(dic_arg['zc_continuation']),
                                    reversed(fund_leg.zc_continuation)):

        idx=Functions.find_idx(contract.fix_dates,d)
        adjusted_cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        structure_price=np.sum(adjusted_cashflows[:,idx+1:]*Pt_T,axis=1)

        fundstop_idxs=[Functions.find_idx(fund_leg.pay_dates, contract.pay_dates[j]) for j in stop_idxs]
        fund_idx1=Functions.find_idx(fund_leg.pay_dates,contract.pay_dates[idx])

        fund_price=np.array([compute_fund_price(fund_cf[j,fund_idx1:fund_idx2],fund_Pt_T[j,:fund_idx2-fund_idx1]) 
                            for j,fund_idx2 in enumerate(fundstop_idxs)])
        
        continuation_value=structure_price -fund_price
        
        basis=MAPPING_BASIS.get(basis_option)(dic_arg['undl'][idx],deg)
        reg=clone(regressor_class)
        reg.fit(basis,continuation_value)
        res.insert(0,reg)

    return res

def compute_stop_idxs_with_undl(contract,regressions:list[KNeighborsRegressor | Ridge],dic_arg:dict,
                                deg:int,include_principal:bool,basis_option='polynomial') ->list[int]:
            
    nbsimu=dic_arg['rates'].shape[1]
    res=[len(contract.pay_dates)-1]*nbsimu
    stop_idxs=[]
    cashflows=contract.compute_cashflows(dic_arg)

    for i,d in enumerate(contract.call_dates):
        idx=Functions.find_idx(contract.fix_dates,d)
        time_to_maturity = contract.paygrid[-1] - contract.paygrid[idx]
        epsilon = 0.00 * np.sqrt(time_to_maturity)
        Exercise_value=dic_arg['zc_exercise'][i]*(include_principal+contract.infine*cashflows[:,idx])
        basis=MAPPING_BASIS.get(basis_option)(dic_arg['undl'][idx],deg)
        #Allow small threshold (numerical noise)
        exercise_threshold = 1 +epsilon
        decision=regressions[i].predict(basis)>Exercise_value*exercise_threshold
        new_stop_idxs=[ j for j,b in enumerate(decision) if b==1 and j not in stop_idxs]
        stop_idxs+=new_stop_idxs
        res=[ int(idx) if j in new_stop_idxs else x for j,x in enumerate(res) ]
        
    return res


    
def compute_price(dic_prep:dict,risky_curve,basis_option:str,regressor_class:KNeighborsRegressor | Ridge):
    """
    Compute price for callable bond or swap.
    """
    
    contract=dic_prep['contract']
    
    # Check if this is a swap or bond
    is_swap='funding_leg' in dic_prep.keys()
    
    if is_swap:
        zc=risky_curve.discount_factor(contract.pay_dates,risky=False)
        funding_leg=dic_prep['funding_leg']
        funding_zc=risky_curve.discount_factor(funding_leg.pay_dates,risky=False)
    else:
        zc=risky_curve.discount_factor(contract.pay_dates,risky=True)
    
    dic_arg=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg']) 
    cashflows=contract.compute_cashflows(dic_arg)

    #Process if callable
    if 'dic_arg_helper' in dic_prep.keys():
        deg=3
        dic_arg_helper=contract.update_arg_pricing(contract.coupon,dic_prep['dic_arg_helper'])
        
        if is_swap:
            regressions=get_regression_for_swap_with_undl(contract,dic_arg_helper,
                                                        funding_leg,contract.funding_spread,deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
            include_principal=False
        else:
            regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
            include_principal=True
        
        stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                            deg,include_principal=include_principal,
                                            basis_option=basis_option)

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
        structure_price=sum(contract.res_coupon*zc)
        funding_price=sum(funding_leg.coupons*funding_zc)
        price=structure_price-funding_price
    else:
        prices=contract.res_coupon+contract.res_capital
        price=sum(prices*zc)
    
    res=dict()
    res['table']=Base.organize_structure_table(contract,zc)
    res['price']=price
    res["duration"]=sum(contract.proba_recall*contract.paygrid)
    res["coupon"]=contract.coupon
    res["funding_spread"]=Base.get_funding_spread_early_redemption(risky_curve,
                                                                contract.pay_dates,contract.proba_recall,
                                                                contract.funding_adjustment)
    
    if is_swap:
        res['funding_table']=Base.organize_funding_table(funding_leg,funding_zc)
    
    return res

def solve_coupon(dic_prep:dict,risky_curve,basis_option:str,regressor_class:KNeighborsRegressor | Ridge):
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
        deg=3
        if is_swap:
            def func_to_solve(x:float):
                dic_arg_helper=contract.update_arg_pricing(x,dic_prep['dic_arg_helper']) 
                dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
                #Compute spread based on bond's duration
                regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
                stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                            deg,include_principal=True,
                                            basis_option=basis_option)
                proba_recall=contract.compute_recall_proba(stop_idxs)
                funding_spread=Base.get_funding_spread_early_redemption(risky_curve,contract.paygrid,
                                                                        proba_recall,
                                                                        contract.funding_adjustment)
                #Use spread to compute swap value
                regressions=get_regression_for_swap_with_undl(contract,dic_arg_helper,
                                                            funding_leg,funding_spread,deg,
                                                            basis_option=basis_option,
                                                            regressor_class=regressor_class)
            
                stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                    deg,include_principal=False,
                                                    basis_option=basis_option)
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
                regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
                dic_arg=contract.update_arg_pricing(x,dic_prep['dic_arg']) 
                stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                        deg,include_principal=True,
                                                        basis_option=basis_option)
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
            regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
            stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                deg,include_principal=True)
            proba_recall=contract.compute_recall_proba(stop_idxs)
            res_funding=Base.get_funding_spread_early_redemption(risky_curve,contract.paygrid,
                                                                    proba_recall,
                                                                    contract.funding_adjustment)
        else:
            regressions=get_regression_for_bond_with_undl(contract,dic_arg_helper,deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
            stop_idxs=compute_stop_idxs_with_undl(contract,regressions,dic_arg,
                                                deg,include_principal=True)
            proba_recall=contract.compute_recall_proba(stop_idxs)
            res_funding=Base.get_funding_spread_early_redemption(risky_curve,contract.paygrid,
                                                                proba_recall,contract.funding_adjustment)

        return res_coupon,res_funding
