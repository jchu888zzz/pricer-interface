import numpy as np
import QuantLib as ql
import scipy
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from Pricing.Utilities import Functions
from Pricing.Rates.Payoffs.Types import Base#,MinMax,Digit,FixedRate,RangeAccrual
from Pricing.Curves.Classic import Risky_Curve
from Pricing.Rates.Payoffs import Funding
from collections import Counter

def prep_discount_factor_from_rates(contract,model,risky_curve
                                    ,dic_undl:dict,risky:bool):
    zc_cont=[None]*len(contract.call_dates)  
    zc_exec=[None]*len(contract.call_dates)
    
    calendar=risky_curve.calendar
    calc_date=risky_curve.calc_date

    for i,d in enumerate(contract.call_dates):
        idx=Functions.find_idx(contract.fix_dates,d)
        t_fix=calendar.yearFraction(calc_date,contract.fix_dates[idx])
        t_cont=np.array([calendar.yearFraction(calc_date,d) for d in contract.pay_dates[idx:]] )
        zc_exec[i]=model.compute_discount_factor_from_rates(dic_undl['rates'][idx],t_fix,
                                                                t_cont[0]).ravel()
        zc_cont[i]=model.compute_discount_factor_from_rates(dic_undl['rates'][idx],t_fix,
                                                                    t_cont[1:])

        if risky:
            zc_exec[i]*=risky_curve.adjustment(t_fix,t_cont[0])
            zc_cont[i]*=np.array([risky_curve.adjustment(t_fix,t) for t in t_cont[1:] ])

    dic_undl['zc_exercise']=zc_exec
    dic_undl['zc_continuation']=zc_cont    
    return dic_undl
    
def precomputation(calc_date:ql.Date,
                   contract,
                   model,risky_curve:Risky_Curve,structure_choice:str="Bond"):
    """
    Precomputation for callable contracts with underlying.

    This function should ONLY be called for callable contracts (contracts with call_dates).
    For bullet contracts, use Base.prep_bullet_contract instead.

    Returns dict with:
        - contract: updated contract with paygrid, fwds, etc.
        - dic_arg: main simulation data for pricing
        - dic_arg_helper: helper simulation data for LSM regression
        - funding_leg: (if swap) funding leg object
    """
    Base.prep_contract_common(calc_date, contract, risky_curve)

    calendar=ql.Actual360()
    data_rates=model.generate_rates(calc_date,contract.pay_dates[-1],
                        cal=calendar,Nbsimu=10000,seed=0)
    data_rates_helper=model.generate_rates(calc_date,contract.pay_dates[-1],cal=calendar,
                                Nbsimu=10000,seed=42)

    if contract.hasunderlying :
        undl_prep = Base.prep_undl(contract, model, data_rates, include_rates=True)
        contract.fwds=np.mean(undl_prep['undl'],axis=1)

        undl_prep_helper=Base.prep_undl(contract,model,data_rates_helper,include_rates=True)

    else:
        #Fixed Rate
        undl_prep =model.compute_prep_for_swaption_from_rates(contract,data_rates,
                                            daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis),
                                            include_rates=True)
        
        undl_prep_helper=model.compute_prep_for_swaption_from_rates(contract,data_rates_helper,
                                        daycount_calendar=ql.Thirty360(ql.Thirty360.BondBasis),
                                        include_rates=True)

    measure_change_factor=np.array([Base.compute_measure_change_factor(model,undl_prep['rates'][i],t,contract.paygrid[-1])
                                    for i,t in enumerate(contract.paygrid) ])[:,:,0]
    contract._measure_change_factor=measure_change_factor

    if structure_choice=='Bond':
        risky=True
        # Callable contract processing - generate helper simulations for LSM
        prep_discount_factor_from_rates(contract,model,risky_curve,undl_prep,risky)
        contract._simu=undl_prep

        prep_discount_factor_from_rates(contract,model,risky_curve,undl_prep_helper,risky)
        contract._simu_helper=undl_prep_helper
        contract.zc=risky_curve.discount_factor(contract.pay_dates,risky)

        return 
    
    if structure_choice=='Swap':
        # Callable contract processing - generate helper simulations for LSM
        risky=False
        prep_discount_factor_from_rates(contract,model,risky_curve,undl_prep,risky)
        contract._simu=undl_prep

        prep_discount_factor_from_rates(contract,model,risky_curve,undl_prep_helper,risky)
        contract._simu_helper=undl_prep_helper
        contract.zc=risky_curve.discount_factor(contract.pay_dates,risky)

        funding_leg=Funding.Leg(contract,contract.currency)
        funding_leg.precomputation(calc_date,model,data_rates)
        funding_leg.zc=risky_curve.discount_factor(funding_leg.pay_dates,risky)

        funding_leg_helper=Funding.Leg(contract,contract.currency)
        funding_leg_helper.precomputation(calc_date,model,data_rates_helper)
        funding_leg_helper.zc=risky_curve.discount_factor(funding_leg_helper.pay_dates,risky)

        contract._funding_leg=funding_leg
        contract._funding_leg_helper=funding_leg

        return 

    raise ValueError(f"{structure_choice} not implemented")

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
                                    regressor_class:KNeighborsRegressor | Ridge,deg:int,
                                    basis_option='polynomial') -> list[KNeighborsRegressor | Ridge]:
    """ Precomputation must be done before"""
    simu_helper=contract._simu_helper
    stop_idxs=[len(contract.pay_dates)-1]*simu_helper['nbsimu']
    
    res=[]
    cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"helper")
    for d,df_cont,df_exercise in zip(reversed(contract.call_dates),
                                    reversed(simu_helper['zc_continuation']),
                                    reversed(simu_helper['zc_exercise'])):
        
        idx=Functions.find_idx(contract.fix_dates,d)
        adjusted_cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        #Continuation value
        continuation_value=np.sum(adjusted_cashflows[:,idx+1:]*df_cont,axis=1)
        #Add capital at stopping time
        continuation_value+= np.array([x[k-(idx+1)] for x,k in zip(df_cont,stop_idxs)])

        basis=MAPPING_BASIS.get(basis_option)(simu_helper['undl'][idx],deg)
        reg=clone(regressor_class)
        reg.fit(basis,continuation_value)
        res.insert(0,reg)
        exercise_value = df_exercise * (1 + contract.infine * cashflows[:, idx])
        prediction = reg.predict(basis)
        decision = prediction>exercise_value

        #Update stop idxs based on decision
        stop_idxs= [idx if d else old_value for d,old_value in zip(decision,stop_idxs) ]

    return res

def get_regression_for_swap_with_undl(contract,dic_arg:dict,spread:float,
                                        regressor_class:KNeighborsRegressor | Ridge,deg:int,
                                        basis_option='polynomial') -> list[KNeighborsRegressor | Ridge]:
    """ Precomputation must be done before"""
    simu_helper=contract._simu_helper
    stop_idxs=[len(contract.pay_dates)-1]*simu_helper['nbsimu']

    def compute_fund_price(cf:np.ndarray,df:np.ndarray):
        if cf.size==0:
            return 0
        else:
            return sum(cf*df)
    res=[]

    cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"helper")
    
    funding_leg=contract._funding_leg_helper
    fund_cf=funding_leg.compute_cashflows(spread)

    for d,Pt_T,fund_Pt_T in zip(reversed(contract.call_dates),
                                    reversed(simu_helper['zc_continuation']),
                                    reversed(funding_leg.zc_continuation)):

        idx=Functions.find_idx(contract.fix_dates,d)
        adjusted_cashflows=Base.adjust_to_stop_idxs(cashflows,stop_idxs,contract.infine)
        structure_price=np.sum(adjusted_cashflows[:,idx+1:]*Pt_T,axis=1)

        fundstop_idxs=[Functions.find_idx(funding_leg.pay_dates, contract.pay_dates[j]) for j in stop_idxs]
        fund_idx1=Functions.find_idx(funding_leg.pay_dates,contract.pay_dates[idx])

        fund_price=np.array([compute_fund_price(fund_cf[j,fund_idx1:fund_idx2],fund_Pt_T[j,:fund_idx2-fund_idx1]) 
                            for j,fund_idx2 in enumerate(fundstop_idxs)])
        
        continuation_value=structure_price -fund_price
        
        basis=MAPPING_BASIS.get(basis_option)(simu_helper['undl'][idx],deg)
        reg=clone(regressor_class)
        reg.fit(basis,continuation_value)
        res.insert(0,reg)

    return res

def compute_stop_idxs_with_undl(contract,dic_arg:dict,include_principal:bool,
                                regressions:list[KNeighborsRegressor | Ridge],
                                deg:int,basis_option='polynomial') ->list[int]:
    """ Precomputation must be done before"""
    simu=contract._simu            
    res=[len(contract.pay_dates)-1]*simu['nbsimu']
    stop_idxs=[]

    cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"classic")
    for i,d in enumerate(contract.call_dates):
        idx=Functions.find_idx(contract.fix_dates,d)
        time_to_maturity = contract.paygrid[-1] - contract.paygrid[idx]
        epsilon = 0.00 * np.sqrt(time_to_maturity)
        Exercise_value=simu['zc_exercise'][i]*(include_principal+contract.infine*cashflows[:,idx])
        basis=MAPPING_BASIS.get(basis_option)(simu['undl'][idx],deg)
        #Allow small threshold (numerical noise)
        exercise_threshold = 1 +epsilon
        decision=regressions[i].predict(basis)>Exercise_value*exercise_threshold
        new_stop_idxs=[ j for j,b in enumerate(decision) if b==1 and j not in stop_idxs]
        stop_idxs+=new_stop_idxs
        res=[ int(idx) if j in new_stop_idxs else x for j,x in enumerate(res) ]
        
    return res

def compute_bond_price(contract,dic_arg:dict=None,deg=3,basis_option='polynomial') -> float:
        """ Precomputation must be done before"""
        dic_arg = Base._validate_and_set_dic_arg(contract,dic_arg)
        cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"classic")
        regressions = get_regression_for_bond_with_undl(contract, dic_arg,
                                                    regressor_class=contract._regressor_class,deg=deg,
                                                    basis_option=basis_option)

        stop_idxs = compute_stop_idxs_with_undl(contract,dic_arg,include_principal=True,
                                                                regressions=regressions,deg=deg,
                                                                basis_option=basis_option)
        cashflows = Base.adjust_to_stop_idxs(cashflows, stop_idxs, contract.infine)
        contract.res_coupon = np.mean(cashflows, axis=0)
        contract.proba_recall = contract.compute_recall_proba(stop_idxs)

        contract.res_capital = Base.compute_bond_measure_change(contract._measure_change_factor, stop_idxs)
        
        prices = contract.res_coupon + contract.res_capital
        res = sum(prices *contract.zc)
        return res

def compute_swap_price(contract,dic_arg:dict=None,deg=3,basis_option='polynomial') -> float:
        """ Precomputation must be done before"""
        dic_arg = Base._validate_and_set_dic_arg(contract,dic_arg)
        cashflows=Base.compute_simulated_cashflows(contract,dic_arg,"classic")
        regressions = get_regression_for_swap_with_undl(contract,dic_arg,contract.funding_spread,
                                                        regressor_class=contract._regressor_class,deg=deg,
                                                        basis_option=basis_option)

        stop_idxs =compute_stop_idxs_with_undl(contract,dic_arg,include_principal=False,
                                                                regressions=regressions,deg=deg,
                                                                basis_option=basis_option)
        contract.proba_recall = contract.compute_recall_proba(stop_idxs)

        cashflows = Base.adjust_to_stop_idxs(cashflows, stop_idxs, contract.infine)
        contract.res_coupon = np.mean(cashflows, axis=0)
        structure_price = sum(contract.res_coupon * contract.zc)
        
        funding_leg = contract._funding_leg
        funding_leg.compute_values_for_early_redemption(stop_idxs, contract.funding_spread)
        funding_price = sum(funding_leg.coupons * funding_leg.zc)
        
        res = structure_price - funding_price

        return res





















def solve_coupon(dic_prep:dict,basis_option:str,regressor_class:KNeighborsRegressor | Ridge,swap:bool,
                 use_memory:bool=False):
    """
    Solve for optimal coupon for callable bond or swap.

    Args:
        dic_prep: Precomputed data dictionary
        basis_option: Basis function type ('polynomial' or 'laguerre')
        regressor_class: Regressor for LSM algorithm
        swap: If True, solve for swap (requires funding_leg in dic_prep)
        use_memory: If True, use the funding spread from the last optimization iteration
                    instead of recomputing after optimization completes.
    """
    contract = dic_prep['contract']
    risky_curve = dic_prep['risky_curve']
    target = contract.UF + contract.yearly_buffer * contract.paygrid[-1]

    if swap:
        if 'funding_leg' not in dic_prep:
            raise ValueError("swap=True requires 'funding_leg' in dic_prep")
        funding_leg = dic_prep['funding_leg']
        funding_ZC = risky_curve.discount_factor(funding_leg.pay_dates, risky=False)
        zc = risky_curve.discount_factor(contract.pay_dates, risky=False)
    else:
        zc = risky_curve.discount_factor(contract.pay_dates, risky=True)

    deg = 3

    # Memory storage for tracking funding spread during optimization
    memory = {'funding_spread': None}

    if swap:
        def func_to_solve(x: float):
            dic_arg_helper = contract.update_arg_pricing(x, dic_prep['dic_arg_helper'])
            dic_arg = contract.update_arg_pricing(x, dic_prep['dic_arg'])
            # Compute spread based on bond's duration
            regressions = get_regression_for_bond_with_undl(contract, dic_arg_helper, deg,
                                                            basis_option=basis_option,
                                                            regressor_class=regressor_class)
            stop_idxs = compute_stop_idxs_with_undl(contract, regressions, dic_arg,
                                                    deg, include_principal=True,
                                                    basis_option=basis_option)
            proba_recall = contract.compute_recall_proba(stop_idxs)
            funding_spread = Base.get_funding_spread_early_redemption(risky_curve, contract.pay_dates,
                                                                        proba_recall,
                                                                        contract.funding_adjustment)
            # Store in memory for later retrieval
            memory['funding_spread'] = funding_spread

            # Use spread to compute swap value
            regressions = get_regression_for_swap_with_undl(contract, dic_arg_helper,
                                                            funding_leg, funding_spread, deg,
                                                            basis_option=basis_option,
                                                            regressor_class=regressor_class)
            stop_idxs = compute_stop_idxs_with_undl(contract, regressions, dic_arg,
                                                    deg, include_principal=False,
                                                    basis_option=basis_option)
            cashflows = contract.compute_cashflows(dic_arg)
            cashflows = Base.adjust_to_stop_idxs(cashflows, stop_idxs, contract.infine)
            coupons = np.mean(cashflows, axis=0)
            structure_price = sum(coupons * zc)

            funding_leg.compute_values_for_early_redemption(stop_idxs, funding_spread)
            funding_price = sum(funding_leg.coupons * funding_ZC)
            return (structure_price - funding_price + target) ** 2
    else:
        def func_to_solve(x: float):
            dic_arg_helper = contract.update_arg_pricing(x, dic_prep['dic_arg_helper'])
            regressions = get_regression_for_bond_with_undl(contract, dic_arg_helper, deg,
                                                            basis_option=basis_option,
                                                            regressor_class=regressor_class)
            dic_arg = contract.update_arg_pricing(x, dic_prep['dic_arg'])
            stop_idxs = compute_stop_idxs_with_undl(contract, regressions, dic_arg,
                                                    deg, include_principal=True,
                                                    basis_option=basis_option)
            # Compute and store funding spread for memory mode
            proba_recall = contract.compute_recall_proba(stop_idxs)
            funding_spread = Base.get_funding_spread_early_redemption(risky_curve, contract.pay_dates,
                                                                        proba_recall,
                                                                        contract.funding_adjustment)
            memory['funding_spread'] = funding_spread

            cashflows = contract.compute_cashflows(dic_arg)
            cashflows = Base.adjust_to_stop_idxs(cashflows, stop_idxs, contract.infine)
            coupons = np.mean(cashflows, axis=0)
            capital = Base.compute_bond_measure_change(dic_arg['measure_change_factor'], stop_idxs)
            return (sum((coupons + capital) * zc) - (1 - target)) ** 2

    res_coupon = Base.optimize_coupon(func_to_solve)

    if use_memory:
        # Use the funding spread from the last optimization iteration
        res_funding = memory['funding_spread']
    else:
        # Recompute funding spread based on optimized coupon (original behavior)
        dic_arg_helper = contract.update_arg_pricing(res_coupon, dic_prep['dic_arg_helper'])
        dic_arg = contract.update_arg_pricing(res_coupon, dic_prep['dic_arg'])
        regressions = get_regression_for_bond_with_undl(contract, dic_arg_helper, deg,
                                                        basis_option=basis_option,
                                                        regressor_class=regressor_class)
        stop_idxs = compute_stop_idxs_with_undl(contract, regressions, dic_arg,
                                                deg, include_principal=True,
                                                basis_option=basis_option)
        proba_recall = contract.compute_recall_proba(stop_idxs)
        res_funding = Base.get_funding_spread_early_redemption(risky_curve, contract.pay_dates,
                                                                proba_recall, contract.funding_adjustment)

    return res_coupon, res_funding

