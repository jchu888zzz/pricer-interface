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




