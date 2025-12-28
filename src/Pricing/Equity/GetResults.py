import numpy as np
import QuantLib as ql

import EQDataPrep
from Pricing.Equity.Model import HestonModel

def get_callprobas(S_fix, call_lvl, S0):
    Nsim, n = S_fix.shape
    probas = np.zeros(n)
    redeemed = []
    for i in range(n - 1):
        idx_call = np.where(S_fix[:,i] >= call_lvl*S0)[0]
        idx_call = np.setdiff1d(idx_call, redeemed)
        probas[i] = len(idx_call)/Nsim
        redeemed = np.union1d(redeemed, idx_call)
    probas[-1] = 1 - np.sum(probas, axis = 0)
    return probas

def geteq_fund2(T_pay, probas, fund_curve, discount):
    c=0.0004
    tenor_spread = fund_curve(T_pay)
    fund2 = (np.sum(tenor_spread*probas, axis = 0) - c)*discount
    return tenor_spread, fund2

def compute_result(input:dict):
    
    _input=EQDataPrep.convert_input(input)
    pay_dates = list(ql.Schedule(_input["start_date"], _input["start_date"] + ql.Period(_input["mat"], ql.Years), ql.Period(_input["freq"]), 
                                 ql.TARGET(), ql.Following, ql.Following, ql.DateGeneration.Forward, False))[1 + _input["per_nocall"]:]
    fix_dates = list(map(lambda date: ql.TARGET().advance(date, _input["offset"], ql.Days), pay_dates))
    T_pay = np.array(list(map(lambda date: ql.Actual365Fixed().yearFraction(_input["value_date"], date), pay_dates)))
    T_fix = np.array(list(map(lambda date: ql.Actual365Fixed().yearFraction(_input["value_date"], date), fix_dates)))
    params = HestonModel.calib_heston(_input["value_date"].ISO(), _input["vol_surface"])

    Xt = HestonModel.diffuse_heston_1D(params, _input["dt"], _input["Nsim"], T_fix[-1], 10)
    idx_fix = np.array(T_fix/_input["dt"], dtype = int)
    S_fix = _input["forward"](T_fix)*Xt[:, idx_fix]
    T_strike = ql.Actual365Fixed().yearFraction(_input["value_date"], _input["trade_date"])
    S0 = _input["forward"](T_strike)
    probas = get_callprobas(S_fix, _input["call_lvl"], S0)

    if _input["start_date"]> _input["value_date"]+ql.Period('2M'):
        discount=0.95
    else:
        discount=1
    duration = np.sum(T_pay*probas, axis = 0)
    new_spreads, new_fund = geteq_fund2(T_pay, probas, _input["fund_curve"], discount)

    res= {"duration": duration,
         "funding_spread": new_fund,
         "table":{"Payment Dates":pay_dates,
                  "Early Redemption Proba":probas,
                  "Model Forward":_input["forward"](T_pay),
                  "Zero Coupon":_input["df"](T_pay)}
        }

    return res