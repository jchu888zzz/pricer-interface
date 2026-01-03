import numpy as np
from bisect import bisect
import QuantLib as ql

from Pricing.Utilities import Functions,Dates
from Pricing.Rates.Payoffs import Base
from collections import Counter

def get_compound_rate(rates:np.ndarray,d1:ql.Date,d2:ql.Date,N=360):

    bdays=ql.TARGET().businessDayList(d1,d2)
    delta=np.diff(bdays)
    dc=ql.Actual360().dayCount(d1,d2)
    res=np.array([(np.prod(1+r*delta/N)-1)*N/dc for r in rates])
    return res

_DIC_CURRENCY={'EUR':'3M','USD':'Overnight'}
class Leg:

    def __init__(self,contract,currency:str):
        self.type_rate=_DIC_CURRENCY[currency]
        self.contract=contract

        self.maturity_date=contract.issue_date+ql.Period(contract.maturity)
        if self.type_rate=='Overnight':
            self.pay_dates,self.fix_dates=Dates.compute_schedule(contract.issue_date,
                                                                 contract.issue_date+ql.Period(contract.maturity),
                                                                 freq=ql.Period('3M'),
                                                                 offset=-2,
                                                                 fixing_type='in arrears')
        else:
            self.pay_dates,self.fix_dates=Dates.compute_schedule(contract.issue_date,
                                                                 contract.issue_date+ql.Period(contract.maturity),
                                                                 freq=ql.Period('3M'),
                                                                 offset=-5,
                                                                 fixing_type='in advance')

    def precomputation(self,calc_date:ql.Date,model,data_rates:dict):
        rates,schedule=data_rates['rates'],data_rates['schedule'][1:]

        cal=ql.Actual360()
        idx=Functions.find_idx(self.fix_dates,calc_date)
        self.fix_dates=self.fix_dates[idx:]
        self.pay_dates=self.pay_dates[idx:]
        
        fixgrid=np.array([max(0,cal.yearFraction(calc_date,x)) for x in self.fix_dates ])
        paygrid=np.array([cal.yearFraction(calc_date,x) for x in self.pay_dates])
        self.delta=np.diff(paygrid,prepend=0)
    
        t_maturity=cal.yearFraction(calc_date,self.maturity_date)
        if self.type_rate=='Overnight':
            self.fwds=np.zeros((len(self.fix_dates),rates.shape[0]))
            start_date=self.fix_dates[0]
            start_idx=Functions.find_idx(schedule,start_date)
            for i,end_date in enumerate(self.fix_dates[1:]):
                end_idx=Functions.find_idx(schedule,end_date)
                self.fwds[i]=get_compound_rate(rates[:,start_idx:end_idx],start_date,end_date,N=360)
                start_date=end_date
                start_idx=end_idx

        elif self.type_rate=='3M':            
            idxs=Functions.find_idx(schedule,self.fix_dates)
            self.fwds=model.compute_deposit_from_rates(rates[:,idxs],fixgrid,'3M')
            #Set first value to spot to avoid numerical approximation at beginning
            h=0.25
            zc=model.curve.parametric_form(h)
            self.fwds[:,0]=(1-zc)/(zc*h)

        else:
            raise ValueError(f'{self.type_rate} not implemented')
        
        contract=self.contract
        if not hasattr(contract,'call_dates'):
            return
        else:
            idxs=[Functions.find_idx(schedule,d) for d in self.pay_dates]
            self.measure_change_factor=Base.compute_measure_change_factor(model,rates[:,idxs],paygrid,t_maturity) 
            
            self.zc_continuation=[None]*len(contract.call_dates)
            for i,d in enumerate(contract.call_dates):
                fix_idx=Functions.find_idx(schedule,d)
                t=cal.yearFraction(calc_date,schedule[fix_idx])
                pay_idx=bisect(self.pay_dates,d)
                self.zc_continuation[i]=model.compute_discount_factor_from_rates(rates[:,idx],t,paygrid[pay_idx:])

    def compute_cashflows(self,spread:float):  
        return (self.fwds+spread)*np.tile(self.delta,(self.fwds.shape[0],1))
    
    def compute_values(self,spread:float):
        self.proba=np.ones(len(self.pay_dates))
        self.coupons=np.mean(self.compute_cashflows(spread),axis=0)

    def compute_values_for_early_redemption(self, stop_idxs: list[int], spread: float):
        contract = self.contract
        
        # Create mask_alive: (num_simulations, num_pay_dates)
        # Marks which coupons are still paid in each simulation path
        mask_alive = np.ones((len(stop_idxs), len(self.pay_dates)))
        for i, idx in enumerate(stop_idxs):
            j = bisect(self.pay_dates, contract.pay_dates[idx])
            mask_alive[i, j:] = 0
        
        # Compute cashflows once: (num_simulations, num_pay_dates)
        cashflows = self.compute_cashflows(spread)
        
        # Initialize result arrays
        self.coupons = np.zeros(len(self.pay_dates))
        self.proba = np.ones(len(self.pay_dates))
        
        # Vectorized approach: separate early and late dates
        first_call_date = contract.call_dates[0]
        pay_dates_array = np.array([d for d in self.pay_dates])
        early_mask = pay_dates_array <= first_call_date
        late_mask = ~early_mask
        
        # Early dates (before first call): simpler calculation
        if np.any(early_mask):
            early_idx = np.where(early_mask)[0]
            self.coupons[early_idx] = np.mean(cashflows[:, early_idx], axis=0)
        
        # Late dates (after first call): vectorized with early redemption logic
        if np.any(late_mask):
            late_idx = np.where(late_mask)[0]
            cf_late = cashflows[:, late_idx]  # (num_sims, num_late)
            mask_late = mask_alive[:, late_idx]  # (num_sims, num_late)
            
            mcf_late = self.measure_change_factor[:, late_idx]  # (num_simulations, num_late)
            
            # Vectorized probability: proportion of simulations alive at each date
            # Using min(1.0, ...) per element to match original logic
            prob_ratios = mask_late / np.maximum(mcf_late, 1e-10)  # Avoid division by zero
            self.proba[late_idx] = np.minimum(1.0, np.mean(prob_ratios, axis=0))
            
            # Vectorized amount: average of (cashflow / measure_change_factor) * alive_mask
            amounts = np.mean((cf_late / mcf_late) * mask_late, axis=0)
            self.coupons[late_idx] = amounts

    